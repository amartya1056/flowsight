# FlowSight — every target delegates to scripts/tasks.py (works on Windows without make, too:
#   python scripts/tasks.py <target>)
PY ?= python

.PHONY: setup data process sim eval test api web build demo bench
setup:   ; $(PY) scripts/tasks.py setup
data:    ; $(PY) scripts/tasks.py data
process: ; $(PY) scripts/tasks.py process
sim:     ; $(PY) scripts/tasks.py sim
eval:    ; $(PY) scripts/tasks.py eval
test:    ; $(PY) scripts/tasks.py test
api:     ; $(PY) scripts/tasks.py api
web:     ; $(PY) scripts/tasks.py web
build:   ; $(PY) scripts/tasks.py build
demo:    ; $(PY) scripts/tasks.py demo
bench:   ; $(PY) scripts/tasks.py bench
