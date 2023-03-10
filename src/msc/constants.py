text_sep_cols = ["image_type",
                 "sequence_variant",
                 "diffusion_bvalue",
                 "in_plane_phase_encoding_direction",
                 "mr_acquisition_type",
                 "patient_position",
                 "manufacturer",
                 "manufacturer_model_name",
                 "receive_coil_name",
                 "transmit_coil_name",
                 "scanning_sequence",
                 "software_versions",
                 "scan_options",
                 "photometric_interpretation",
                 "modality",
                 "series_description"]

num_sep_cols = ["acquisition_matrix",
                "echo_time",
                "echo_train_length",
                "repetition_time",
                "flip_angle",
                "reconstruction_matrix",
                "magnetic_field_strength",
                "number_of_phase_encoding_steps",
                "percent_phase_field_of_view",
                "pixel_bandwidth",
                "sar",
                "slice_thickness",
                "temporal_resolution",
                "image_orientation_patient",
                "inversion_time",
                "pixel_spacing",
                "number_of_echos",
                "number_of_temporal_positions"]

num_cols = ["number_of_images"]

cols_to_drop = ["study_uid","series_uid","class","diffusion_directionality",
                "spectrally_selected_suppression","reconstruction_matrix",
                "number_of_phase_encoding_steps","inversion_time"]

replace_cols = {"diffusion_bvalue":"-",
                "number_of_echos":"1",
                "number_of_temporal_positions":"1",
                "temporal_resolution":"0"}