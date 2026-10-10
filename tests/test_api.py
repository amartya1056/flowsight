"""Phase-7: hit every endpoint with a short simulator run (TestClient)."""
import time

import pytest
from fastapi.testclient import TestClient

from api.main import RUNS, app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def run_id(client):
    r = client.post("/api/runs", json={"store_id": "demo_store", "source": "sim:rush_hour", "speed": "max",
                                       "duration_s": 420, "seed": 1})
    assert r.status_code == 200, r.text
    rid = r.json()["run_id"]
    t0 = time.time()
    while not RUNS[rid].p.finished and time.time() - t0 < 120:
        time.sleep(0.25)
    assert RUNS[rid].p.finished and RUNS[rid].p.error is None
    return rid


def test_health_and_stores(client):
    assert client.get("/api/health").json()["ok"]
    stores = client.get("/api/stores").json()
    assert any(s["store_id"] == "demo_store" for s in stores)
    d = client.get("/api/stores/demo_store").json()
    assert d["zones"] and d["floorplan_url"]
    r = client.get("/api/stores/demo_store/floorplan.png")
    assert r.status_code == 200 and r.content[:4] == b"\x89PNG"
    assert client.get("/api/stores/nope").status_code == 404


def test_run_lifecycle_and_reads(client, run_id):
    assert any(r["run_id"] == run_id for r in client.get("/api/runs").json())
    info = client.get(f"/api/runs/{run_id}").json()
    assert info["finished"] and info["progress"] == 1.0
    k = client.get(f"/api/runs/{run_id}/kpis").json()
    assert k["visitors_total"] > 5
    zones = client.get(f"/api/runs/{run_id}/zones").json()
    assert len(zones) == 14 and "funnel" in zones[0]
    png = client.get(f"/api/runs/{run_id}/heatmap?type=occupancy")
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"
    g = client.get(f"/api/runs/{run_id}/heatmap?type=stationary&window=300&format=json").json()
    assert g["nx"] == 80 and g["ny"] == 48 and g["total_person_s"] > 0
    assert client.get(f"/api/runs/{run_id}/heatmap?type=bad").status_code == 400
    flow = client.get(f"/api/runs/{run_id}/flow").json()
    assert flow["sankey"]["nodes"] and flow["top_paths"]
    assert isinstance(client.get(f"/api/runs/{run_id}/bottlenecks").json(), list)
    att = client.get(f"/api/runs/{run_id}/attention").json()
    assert att["zones"] and "what_if" in att
    q = client.get(f"/api/runs/{run_id}/queue").json()
    assert q[0]["zone_id"] == "z_checkout"
    lay = client.get(f"/api/runs/{run_id}/layout").json()
    assert {"what_if", "recurring_bottlenecks", "ignored", "unvisited", "hot"} <= set(lay)


def test_insights_ack_briefing_exports(client, run_id, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    allins = client.get(f"/api/runs/{run_id}/insights?status=all").json()
    assert allins, "rush hour should produce at least one insight"
    active = client.get(f"/api/runs/{run_id}/insights").json()
    assert len(active) <= 12
    target = (active or allins)[0]
    r = client.post(f"/api/runs/{run_id}/insights/{target['id']}/ack")
    assert r.status_code == 200
    assert client.post(f"/api/runs/{run_id}/insights/nope/ack").status_code == 404
    snap = next((i["evidence"]["snapshot_path"] for i in allins if i["evidence"]["snapshot_path"]), None)
    if snap:
        assert client.get(f"/api/runs/{run_id}/{snap}").status_code == 200
    b = client.get(f"/api/runs/{run_id}/briefing").json()
    assert b["mode"] == "template" and len(b["bullets"]) == 5
    csv = client.get(f"/api/runs/{run_id}/export.csv")
    assert csv.status_code == 200 and "zone_id" in csv.text.splitlines()[0]
    html = client.get(f"/api/runs/{run_id}/report.html")
    assert html.status_code == 200 and "FlowSight" in html.text


def test_mjpeg_stream_returns_frames(client, run_id):
    with client.stream("GET", f"/api/runs/{run_id}/stream.mjpg?max_frames=1") as r:
        assert r.headers["content-type"].startswith("multipart/x-mixed-replace")
        body = b"".join(r.iter_bytes())
    assert b"--frame" in body and b"\xff\xd8" in body  # JPEG SOI
    assert client.get(f"/api/runs/{run_id}/frame.jpg").status_code == 200


def test_websocket_emits(client, run_id):
    with client.websocket_connect(f"/api/runs/{run_id}/live") as ws:
        m = ws.receive_json()
    assert {"ts", "people", "zones", "insights_delta", "kpis"} <= set(m)
    assert len(m["zones"]) == 14


def test_live_run_stream_and_stop(client):
    r = client.post("/api/runs", json={"store_id": "demo_store", "source": "sim:baseline", "speed": 2, "duration_s": 600})
    rid = r.json()["run_id"]
    t0 = time.time()
    while RUNS[rid].p.latest_jpeg is None and time.time() - t0 < 20:
        time.sleep(0.1)
    with client.websocket_connect(f"/api/runs/{rid}/live") as ws:
        assert "people" in ws.receive_json()
    assert client.delete(f"/api/runs/{rid}").json()["finished"]


def test_zone_and_calibration_edits(client, tmp_path, monkeypatch):
    import flowsight.config as fc

    saved = {}
    monkeypatch.setattr("api.main.save_store", lambda st: saved.setdefault("st", st))
    zones = client.get("/api/stores/demo_store").json()["zones"]
    r = client.put("/api/stores/demo_store/zones", json={"zones": zones[:3]})
    assert r.json() == {"ok": True, "zones": 3} and len(saved["st"].zones) == 3
    body = {"camera_id": "cam1", "image_points": [[0, 0], [1000, 0], [1000, 600], [0, 600]],
            "floor_points": [[0, 0], [20, 0], [20, 12], [0, 12]], "save": False}
    r = client.put("/api/stores/demo_store/calibration", json=body).json()
    assert r["reprojection_error"]["floor_m_max"] < 1e-6
    bad = dict(body, image_points=[[0, 0], [1, 1]], floor_points=[[0, 0], [1, 1]])
    assert client.put("/api/stores/demo_store/calibration", json=bad).status_code == 400
    assert fc.STORES_DIR.exists()


def test_multicam_store_offers_fusion(client):
    stores = {s["store_id"]: s for s in client.get("/api/stores").json()}
    assert stores["caviar_multicam"]["multi"] is True
    assert stores["demo_store"]["multi"] is False
    r = client.post("/api/runs", json={"store_id": "demo_store", "source": "multi", "speed": "max"})
    assert r.status_code == 400


def test_store_detail_keeps_camera_calibration(client):
    cams = client.get("/api/stores/caviar_corridor").json()["cameras"]
    assert cams[0]["calibration"]["image_points"] and cams[0]["available"] in (True, False)
