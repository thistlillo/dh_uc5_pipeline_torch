# UC5 "Deep Image Annotation", DeepHealth
# 
# Makefile for the PyTorch-based implementation, to be run on HPC4AI premises
#   the entire UC5 pipeline can be run via this makefile
#
# author: Franco Alberto Cardillo <francoalberto.cardillo@ilc.cnr.it>
# 2021-07-31: First (published) release
# 
# Notes:
# 	- in this makefile, normal (info) messages are print using ($warning <msg>) to output also the line number corresponding to the message
#

# V A R I A B L E S
PYTHON ::= python3
$(warning python interpreter defined as ${PYTHON})

# BASE_DS_FLD: base folder with the data, if "UC5_DATA" is set in the environment, use its value
BASE_DS_FLD ::= $(UC5_DATA)
ifdef UC5_DATA
$(warning UC5_DATA environment variable set)
BASE_DS_FLD = $(UC5_DATA)
else
$(warning no UC5_DATA environment variable, using default)
# 
BASE_DS_FLD = /mnt/datasets/uc5/std-dataset_TEST
BASE_OUT_FLD = /opt/uc5/results/uc5_pipeline_torch_TEST

endif
$(warning ** IMPORTANT ** UC5 data folder: ${BASE_DS_FLD})

TEXT_SUBFLD ::= text
IMAGE_SUBFLD ::= image

IMAGE_FLD ::= ${BASE_DS_FLD}/$(IMAGE_SUBFLD)
TEXT_FLD ::= $(BASE_DS_FLD)/$(TEXT_SUBFLD)



# *
# * D O W N L O A D
# *
$(TEXT_FLD):
	mkdir -p $(BASE_DS_FLD) && cd $(BASE_DS_FLD) && wget https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz && \
	cd $(BASE_DS_FLD) && tar xf NLMCXR_reports.tgz && mv ecgen-radiology $(TEXT_SUBFLD)

$(IMAGE_FLD):
	mkdir -p $(BASE_DS_FLD) && cd $(BASE_DS_FLD) && wget https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz && \
	cd $(BASE_DS_FLD) && mkdir $(IMAGE_SUBFLD) && mv NLMCXR_png.tgz $(IMAGE_SUBFLD) && cd $(IMAGE_SUBFLD) && tar xf NLMCXR_png.tgz && mv NLMCXR_png.tgz ..

#$(addsuffix CXR3708_IM-1852-2001.png, ${IMAGE_FLD})
download: | $(TEXT_FLD) $(IMAGE_FLD)
	$(warning data downloaded and ready to be used - downloaded tgz files in ${BASE_DS_FLD})

clean_downl_tars:
	rm -fr $(addprefix ${BASE_DS_FLD}/, NLMCXR_reports.tgz NLMCXR_png.tgz ecgen-radiology)


# *
# *  P I P E L I N E   A: process raw data (the ones downloaded with "make download")
# * prepare data to be used in experiment - output saved in BASE_OUT_FLD and used by all of the subsequent experimentations
# *

# *** step A.00: from raw data to initial tsv files with 'dirty' data
TSV_FLD ::= ${BASE_OUT_FLD}/tsv
REPORTS_TSV ::= ${TSV_FLD}/reports.tsv

RAW_REPORTS ::= $(TSV_FLD)/reports_raw.tsv
RAW_IMAGES ::= $(TSV_FLD})/images_raw.tsv
TSV_RAW_FILES ::= $(RAW_REPORTS) $(RAW_IMAGES)
# $(warning step A.00 produces: ${TSV_RAW_FILES})
STEP_A00_WIT_RAW_TSVS ::= $(BASE_OUT_FLD)/.A00_witness
# $(warning step A.00 produces multiple outputs: using a witness located at ${STEP_A00_WIT_RAW_TSVS})
$(STEP_A00_WIT_RAW_TSVS): $(TEXT_FLD) $(IMAGE_FLD) A00_process_raw_dataset.py
	$(warning step A_00 produces: ${TSV_RAW_FILES})
	$(warning step A_00 produces multiple outputs: using a witness located at $@)
	@mkdir -p $(BASE_OUT_FLD) && mkdir -p $(TSV_FLD)
	@rm -f $@.tmp
	@touch $@.tmp
	$(PYTHON) A00_process_raw_dataset.py --in_txt_fld=$(TEXT_FLD) --in_img_fld=$(IMAGE_FLD) --out_fld=$(TSV_FLD) --log_level=debug --verbose=False
	@mv -f $@.tmp $@

# *** step A.01: clean tsv files produced by step A.00
VERIFIED_TSV_FILE ::= $(TSV_FLD)/reports_ver.tsv
LOWERCASE_TSV_FILE ::= $(TSV_FLD)/reports_lower.tsv
FINAL_TSV_FILE ::= $(TSV_FLD)/reports.tsv
# this step writes three files. using a witness to keep track of the processing
CLEAN_TSV_FILES ::= $(VERIFIED_TSV_FILE) $(LOWERCASE_TSV_FILE) $(FINAL_TSV_FILE)
STEP_A01_WIT_FINAL_TSV ::= $(BASE_OUT_FLD)/.A01_witness
$(STEP_A01_WIT_FINAL_TSV): $(STEP_A00_WIT_RAW_TSVS) A01_clean_files.py
	$(warning step A_01, A01_clean_files.py, produces: ${CLEAN_TSV_FILES})
	$(warning step A_01 produces multiple outputs: using a witness located at $@)
	@rm -f $@.tmp
	@touch $@.tmp
	$(PYTHON) A01_clean_files.py --in_img_fld=$(IMAGE_FLD) --in_tsv_fld=$(TSV_FLD) --out_fld=$(TSV_FLD) --log_level=debug
	@mv -f $@.tmp $@

# *** step A.2: one-hot encoding and other data-encoding operations
ENCODED_MESH_TERMS ::= $(TSV_FLD)/e_mesh_terms.tsv
ENCODED_IMAGE_LABELS ::= $(TSV_FLD)/e_image_labels.tsv
ENCODED_FILES ::= $(ENCODED_MESH_TERMS) $(ENCODED_IMAGE_LABELS)
STEP_A02_WIT_ENC_MESH ::= $(BASE_OUT_FLD)/.A02_witness
$(STEP_A02_WIT_ENC_MESH) : $(FINAL_TSV_FILE) A02_encode_data.py
	$(warning step A_02, A02_encode_data.py, produces: ${ENCODED_FILES})
	$(warning step A_02 produces multiple outputs: using a witness located at $@)
	@rm -f $@.tmp
	@touch $@.tmp
	$(PYTHON) A02_encode_data.py --in_tsv_fld=$(TSV_FLD) --out_fld=$(TSV_FLD)
	@mv -f $@.tmp $@

# *** step A.3: adjust one-hot image encoding for using a softmax last layer in CNN
ADJUSTED_ENC ::= $(TSV_FLD)/e_image_labels_norm_sm.tsv
$(ADJUSTED_ENC): $(ENCODED_IMAGE_LABELS) A03_adjust_ds_for_softmax.py
	$(warning step A_03 produces ${ADJUSTED_ENC})
	$(PYTHON) A03_adjust_ds_for_softmax.py --in_tsv_fld=$(TSV_FLD) --out_fld=$(TSV_FLD)
	$(warning this step completes the preprocessing "pipeline A")

# aliases
raw_tsv_files : $(STEP_A00_WIT_RAW_TSVS)
final_tsv_file : $(STEP_A01_WIT_FINAL_TSV)
enc_mesh_terms : $(STEP_A02_WIT_ENC_MESH)
alt_enc_images: $(ADJUSTED_ENC)


A_pipeline: | raw_tsv_files final_tsv_file enc_mesh_terms alt_enc_images


# *
# *  P I P E L I N E   B: prepare training data
# * data for the single experimentation, train-val-test, train_mode, training itself, etc
# * Output in OUT_FLD, not in BASE_OUT_FLD
# *

OUT_FLD ::= $(BASE_OUT_FLD)/development
$(warning output folder set to ${OUT_FLD})

RND_SEED ::= 2
SHUFFLE_RND_SEED ::= 7

ifeq ($(RND_SEED),0)
RND_SEED ::= $(shell date +%s)
$(warning random seed set to current time in ms: ${RND_SEED})
endif

ifeq ($(SHUFFLE_RND_SEED),0)
SHUFFLE_RND_SEED ::= $(shell date +%s)
$(warning shuffling random seed set to current time in ms: ${RND_SEED})
endif

TRAIN_P ::= 0.7
VALID_P ::= 0.1
N_FOLDERS ::= 1
# TRAIN_MODE one of random | standard | cross_validation (see common/defaults.py)
TRAIN_MODE ::= standard

# Establish here what encoding is to be used, if 1-hot (e_image_labels.tsv) or 1-hot/L1 (e_image_labels_norm_sm.tsv)
# NOTE: image labels could be ancoded on-the-fly by the dataset loader. We adopted this solution in order to share the
#   dataset between the PyTorch and the PyEDDL implementations.
ALT_ENCODING = 1
ifdef ALT_ENCODING
IMG_ENC = e_image_labels_norm_sm.tsv
else
IMG_ENC = e_image_labels.tsv
endif
IMG_ENC := $(TSV_FLD)/$(IMG_ENC)
$(warning ** IMPORTANT** using image encoding in: ${IMG_ENC})


# *** STEP B_00: prepare folders for the experiment and prepare the split
EXP_NAME ::= dev_exp_mk
EXP_FLD ::= $(OUT_FLD)/$(EXP_NAME)/$(TRAIN_MODE)_$(SHUFFLE_RND_SEED)_$(RND_SEED)

# *** RESULTS_FLD NOT USED ANY MORE: REMOVE REFERENCES IN THIS MAKEFILE
# RESULTS_FLD ::= $(addprefix ${EXP_FLD}, results/ )
# BEST_CNN ::= $(addprefix ${RESULTS_FLD}, onnx/best.onnx)
$(warning experiment folder: ${EXP_FLD})
$(warning creating folder ${EXP_FLD} (if it does not exists) )
$(shell mkdir -p $(EXP_FLD) )
#$(warning results folder: ${RESULTS_FLD})
#$(warning creating folder ${RESULTS_FLD} (if it does not exists) )
#$(shell mkdir -p $(RESULTS_FLD) )

$(warning training mode "${TRAIN_MODE}", train_p ${TRAIN_P}, valid_p ${VALID_P})
ifeq (${TRAIN_MODE},cross_validation)
$(warning ${K_FOLDERS} subfolders will be created in ${EXP_FLD})
else
$(warning ONE subfolder will be created in ${EXP_FLD})
endif

SPLIT_NAMES_I ::= train_i valid_i test_i
SPLIT_NAMES_R ::= train_r valid_r test_r
# SPLIT_INDEXES_[I|R] used only in log messages: they are neither command line args nor targets
SPLIT_INDEXES_I ::= $(addsuffix .tsv, ${addprefix $(EXP_FLD), $(SPLIT_NAMES_I)} )
SPLIT_INDEXES_R ::= $(addsuffix .tsv, ${addprefix $(EXP_FLD), $(SPLIT_NAMES_R)})
STEP_B00_WIT_SPLIT ::= $(EXP_FLD)/.B00_witness
$(warning ${STEP_B00_WIT_SPLIT})

$(STEP_B00_WIT_SPLIT): $(STEP_A01_WIT_FINAL_TSV) B00_prepare_training_data.py
	$(warning step B_00 produces: ${SPLIT_INDEXES_I} ${SPLIT_INDEXES_R} in subfolders of ${EXP_FLD} whose number depends on the training mode)
	$(warning step B_00 produces multiple outputs: using a witness located at $@)
	@rm -f $@.tmp
	@touch $@.tmp
	$(PYTHON) B00_prepare_training_data.py --tsv_fld=$(TSV_FLD) --out_fld=$(EXP_FLD) --image_enc=$(IMG_ENC) --train_mode=$(TRAIN_MODE) --n_folders=$(N_FOLDERS) --random_seed=$(SHUFFLE_RND_SEED) --train_p=$(TRAIN_P) --valid_p=$(VALID_P)
	@mv -f $@.tmp $@

# aliases
prepare_training_data : $(STEP_B00_WIT_SPLIT)


B_pipeline: | prepare_training_data

#  E X T E R N A L  T A R G E T S

all: | A_pipeline B_pipeline
	@echo make all completed

# delete witnesses

clean_wit_A:
	$(warning deleting all witness files in pipeline A)
	@rm -f $(STEP_A00_WIT_RAW_TSVS) $(STEP_A01_WIT_FINAL_TSV)  $(STEP_A02_WIT_ENC_MESH)

clean_wit_B:
	$(warning deleting all witness files in pipeline B)
	@rm -f $(STEP_B00_WIT_SPLIT)

clean_wit: | clean_wit_A clean_wit_B
	$(warning deleting all witness files - all pipelines)

.DEFAULT_GOAL := all
.PHONY: clean_downl_tars download clean_witnesses