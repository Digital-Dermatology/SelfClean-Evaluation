.DEFAULT_GOAL := help

###########################
# HELP
###########################
include *.mk

###########################
# VARIABLES
###########################
PROJECTNAME := selfclean_evaluation
GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
GIT_BRANCH := $(if $(GIT_BRANCH),$(GIT_BRANCH),main)
PROJECT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/)

# docker
GPU_ID := 0

# check if `netstat` is installed
ifeq (, $(shell which netstat))
$(error "Netstat executable not found, install it with `apt-get install net-tools`")
endif

# Check if Jupyter Port is already use and define an alternative
ifeq ($(origin PORT), undefined)
  PORT_USED = $(shell netstat -tl | grep -E '(tcp|tcp6)' | grep -Eo '8888' | tail -n 1)
  # Will fail if both ports 9999 and 10000 are used, I am sorry for that
  NEXT_TCP_PORT = $(shell netstat -tl | grep -E '(tcp|tcp6)' | grep -Eo '[0-9]{4}' | sort | tail -n 1 | xargs -I '{}' expr {} + 1)
  ifeq ($(PORT_USED), 8888)
    PORT = $(NEXT_TCP_PORT)
  else
    PORT = 8888
  endif
endif

# List of directories to potentially map as Docker volumes
PATHS := /media/ /raid/dataset/ /raid/fabian/
DOCKER_PATHS :=
DOCKER_PATHS := $(foreach dir,$(PATHS),$(shell if [ -d $(dir) ]; then echo "$(DOCKER_PATHS) -v $(dir):$(dir)"; else echo "$(DOCKER_PATHS)"; fi))

ENV_FILE := $(shell if [ .env ]; then echo "--env-file .env"; else echo ""; fi)

DOCKER_CMD := docker run \
              -it \
              --rm \
              -v $$PWD:/workspace/ \
              $(DOCKER_PATHS) \
              --name $(PROJECTNAME)_no_gpu \
              --shm-size 8G \
              $(ENV_FILE) \
              $(PROJECTNAME):$(GIT_BRANCH)
DOCKER_GPU_CMD := docker run \
                  -it \
                  --rm \
                  -p $(PORT):8888 \
                  -v $$PWD:/workspace/ \
                  $(DOCKER_PATHS) \
                  --gpus='"device=$(GPU_ID)"' \
                  --name $(PROJECTNAME)_gpu_$(GPU_ID) \
                  --shm-size 100G \
                  $(ENV_FILE) \
                  $(PROJECTNAME):$(GIT_BRANCH)
DOCKER_DGX := docker run \
			  -it \
			  -d \
              -u $(id -u):$(id -g) \
			  -v ${PWD}:/workspace/ \
              $(DOCKER_PATHS) \
			  -w /workspace \
			  --gpus='"device=0,1,2,3"' \
              --name $(PROJECTNAME)_multi_gpu \
			  --shm-size 200G \
              $(ENV_FILE)
TORCH_CMD := torchrun --nnodes 1 --node_rank 0 --nproc_per_node 4
# SSH
PORT := 22
USERNAME := fgroger
DEST_FOLDER := selfclean_evaluation

###########################
# COMMANDS
###########################
# Thanks to: https://stackoverflow.com/a/10858332
# Check that given variables are set and all have non-empty values,
# die with an error otherwise.
#
# Params:
#   1. Variable name(s) to test.
#   2. (optional) Error message to print.
check_defined = \
    $(strip $(foreach 1,$1, \
        $(call __check_defined,$1,$(strip $(value 2)))))
__check_defined = \
    $(if $(value $1),, \
      $(error Undefined $1$(if $2, ($2))))

###########################
# SSH UTILS
###########################
.PHONY: push_ssh
push_ssh: clean  ##@SSH pushes all the directories along with the files to a remote SSH server
	$(call check_defined, SSH_CONN)
	rsync -r --exclude='data/' --exclude='.git/' --exclude='.github/' --exclude='wandb/' --exclude='assets/' --progress -e 'ssh -p $(PORT)' $(PROJECT_DIR)/ $(USERNAME)@$(SSH_CONN):$(DEST_FOLDER)/

.PHONY: push_all_ssh
push_all_ssh: clean  ##@SSH pushes all the directories along with the files to a remote SSH server
	$(call check_defined, SSH_CONN)
	rsync -r --exclude='.git/' --exclude='.github/' --exclude='wandb/' --progress -e 'ssh -p $(PORT)' $(PROJECT_DIR)/ $(USERNAME)@$(SSH_CONN):$(DEST_FOLDER)/

.PHONY: pull_ssh
pull_ssh:  ##@SSH pulls directories from a remote SSH server
	$(call check_defined, SSH_CONN)
	scp -r -P $(PORT) $(USERNAME)@$(SSH_CONN):$(DEST_FOLDER) .

###########################
# PROJECT UTILS
###########################
.PHONY: init
init:  ##@Utils initializes the project and pulls all the nessecary data
	@git submodule update --init --recursive

.PHONY: update_data_ref
update_data_ref:  ##@Utils updates the reference to the submodule to its latest commit
	@git submodule update --remote --merge

.PHONY: clean
clean:  ##@Utils clean the project
	@find . -name '*.pyc' -delete
	@find . -name '__pycache__' -type d | xargs rm -fr
	@rm -f .DS_Store
	@rm -f .coverage coverage.xml report.xml
	@rm -f -R .pytest_cache
	@rm -f -R .idea
	@rm -f -R tmp/
	@rm -f -R cov_html/

.PHONY: install
install:  ##@Utils install the dependencies for the project
	python3 -m pip install -r requirements.txt

###########################
# DOCKER
###########################
_build:
	@echo "Build image $(GIT_BRANCH)..."
	@docker build -f Dockerfile -t $(PROJECTNAME):$(GIT_BRANCH) .

run_bash: _build  ##@Docker run an interactive bash inside the docker image
	@echo "Run inside docker image"
	@-docker rm $(PROJECTNAME)_no_gpu
	$(DOCKER_CMD) /bin/bash

run_gpu_bash: _build  ##@Docker runs an interacitve bash inside the docker image with a GPU
	@echo "Run inside docker image"
	@-docker rm $(PROJECTNAME)_gpu_$(GPU_ID)
	$(DOCKER_GPU_CMD) /bin/bash

start_jupyter: _build  ##@Docker start a jupyter notebook inside the docker image (default: GPU=true)
	@echo "Starting jupyter notebook"
	@-docker rm $(PROJECTNAME)_gpu_$(GPU_ID)
	$(DOCKER_GPU_CMD) /bin/bash -c "jupyter notebook --allow-root --ip 0.0.0.0 --port 8888"

###########################
# TESTS
###########################
.PHONY: test
test: _build  ##@Test run all tests in the project
    # Ignore integration tests flag: --ignore=test/manual_integration_tests/
	$(DOCKER_CMD) /bin/bash -c "wandb offline && python -m pytest --cov-report html:cov_html --cov-report term --cov=src --cov-report xml --junitxml=report.xml ./ && coverage xml"
