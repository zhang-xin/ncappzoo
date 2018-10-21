
ifneq ($(findstring movidius, $(PYTHONPATH)), movidius)
	export PYTHONPATH:=/opt/movidius/caffe/python:$(PYTHONPATH)
endif


ONNX_MODEL = mnist.tar.gz
ONNX_FOLDER = mnist

.PHONY: all
all: deps

.PHONY: data
data:
	@echo "\nmaking data - no data needed."

.PHONY: deps
deps:
	@echo "\nmaking deps"
	wget -c --no-cache -P . https://www.cntk.ai/OnnxModels/mnist/opset_1/mnist.tar.gz
	tar xf mnist.tar.gz
	mo_onnx.py --data_type FP16 --input_model mnist/model.onnx

.PHONY: run
run: deps
	@echo "\nmaking run"
	python3 ./touchcalc.py

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - makes everything needed to run but doesn't run";
	@echo "  make clean - removes all created content in this directory."
	@echo "  make data - downloads data as needed"
	@echo "  make deps - makes/prepares dependencies"
	@echo "  make run - runs the touchcalc.py python example program";

.PHONY: clean
clean: 
	@echo "\nmaking clean"
	rm -f ${ONNX_MODEL}
	rm -rf ${ONNX_FOLDER}
	rm -f model.bin
	rm -f model.mapping
	rm -f model.xml


