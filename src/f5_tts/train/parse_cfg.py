def parse_ppg_config(ppg_cfg):
    transformer_ppg_config = dict(
        use_ppg = True,
        ppg_dim = ppg_cfg.dim
    )
    cfm_ppg_config = dict(
        use_ppg = True,
        combined_cond_drop_prob = ppg_cfg.combined_cond_drop_prob
    )
    trainer_ppg_config = dict(
        use_ppg = True,
        model_path = ppg_cfg.model_path,
        config = ppg_cfg.config,
        frame_length = ppg_cfg.frame_length,
        mel_frame_shift = ppg_cfg.mel_frame_shift,
        dim = ppg_cfg.dim,
        output_type = ppg_cfg.output_type,
        map_mix_ratio = ppg_cfg.map.map_mix_ratio,
        global_phn_center_path = ppg_cfg.map.global_phn_center_path,
        para_softmax_path = ppg_cfg.map.para_softmax_path
    )
    return transformer_ppg_config, cfm_ppg_config, trainer_ppg_config

def parse_codebook_config(codebook_cfg):
    transformer_codebook_config = dict(
        use_codebook = True,
        codebook_prob = codebook_cfg.codebook_prob,
        codebook_loss_weight = codebook_cfg.codebook_loss_weight,
        num_vars = codebook_cfg.num_vars,
        temp_start = codebook_cfg.temp_start,
        temp_stop = codebook_cfg.temp_stop,
        temp_decay = codebook_cfg.temp_decay,
        groups = codebook_cfg.groups,
        combine_groups = codebook_cfg.combine_groups,
        weight_proj_depth = codebook_cfg.weight_proj_depth,
        weight_proj_factor = codebook_cfg.weight_proj_factor
    )
    cfm_codebook_config = dict(
        use_codebook = True,
    )
    return transformer_codebook_config, cfm_codebook_config

def parse_durpred_config(durpred_cfg):
    transformer_durpred_config = dict(
        use_durpred = True,
        style_vector_dim = durpred_cfg.style_vector_dim,
        filter_channels = durpred_cfg.filter_channels,
        kernel_size = durpred_cfg.kernel_size,
        dropout = durpred_cfg.dropout
    )
    cfm_codebook_config = dict(
        use_durpred = True,
    )
    return transformer_durpred_config, cfm_codebook_config
