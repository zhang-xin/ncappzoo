NCCOMPILE = mo_tf.py

MODEL_TAR = ssd_mobilenet_v2_coco_2018_03_29.tar.gz
MODEL_DIR = ssd_mobilenet_v2_coco_2018_03_29
IE_FILENAME_BASE = frozen_inference_graph

GET_MODEL = wget -P . http://download.tensorflow.org/models/object_detection/${MODEL_TAR}

.PHONY: all
all: deps data compile_model

.PHONY: model
model:
	@echo "\ndownloading model"; \
	if [ -e ${MODEL_TAR} ] ; \
	then \
		echo "model file already exists, skipping download."; \
	else \
		echo "Downloading model file"; \
		${GET_MODEL}; \
		if [ -e ${MODEL_TAR} ] ; \
		then \
			echo "model file downloaded."; \
			tar xf ${MODEL_TAR}; \
		else \
			echo "***\nError - Could not download model file. Check network and proxy settings \n***\n"; \
			exit 1; \
		fi ; \
	fi

.PHONY: compile_model
compile_model: model
	@echo "\nmaking compile"; \
	if [ -e ${IE_FILENAME_BASE}.xml ] ; \
	then \
		echo "compiled model file already exists, skipping compile."; \
	else \
		${NCCOMPILE} --input_model ./${MODEL_DIR}/${IE_FILENAME_BASE}.pb --tensorflow_use_custom_operations_config ${INTEL_CVSDK_DIR}/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --data_type FP16 --tensorflow_object_detection_api_pipeline_config ./${MODEL_DIR}/pipeline.config; \
	fi

.PHONY: deps
deps: model
	@echo "\nmaking deps"

.PHONY: data
data: 
	@echo "\nmaking data - no data required for this project"

.PHONY: check
check: 
	@echo "\nCheck not implemented yet."


.PHONY: compile_cpp
compile_cpp: 
	@echo "\ncompile_cpp not implemented yet."

.PHONY: run
run: deps compile_model compile_cpp
	@echo "\nrun not implemented yet."

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - make all - everything needed to run, but doesn't run";
	@echo "  make data - downloads required data.";
	@echo "  make deps - makes dependencies for project, prepares model etc.";
	@echo "  make model - downloads the model for the network"
	@echo "  make compile_model - runs model compiler for the network";
	@echo "  make run - builds and runs the C++ example program";
	@echo "  make clean - removes all created content"

.PHONY: clean
clean: 
	@echo "\nmaking clean"
	rm -f ${MODEL_TAR}
	rm -rf ${MODEL_DIR}
	rm -f ${IE_FILENAME_BASE}.bin
	rm -f ${IE_FILENAME_BASE}.mapping.json
	rm -f ${IE_FILENAME_BASE}.mapping
	rm -f ${IE_FILENAME_BASE}.xml



