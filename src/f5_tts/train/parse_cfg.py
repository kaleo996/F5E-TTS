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
    use_perplex_loss = codebook_cfg.get('use_perplex_loss', False)
    if use_perplex_loss:
        perplex_loss_config = dict(
            perplex_loss_prob = codebook_cfg.perplex_loss_config.perplex_loss_prob,
            perplex_loss_weight = codebook_cfg.perplex_loss_config.perplex_loss_weight
        )
    else:
        perplex_loss_config = dict()
    
    use_align_loss = codebook_cfg.get('use_align_loss', False)
    if use_align_loss:
        align_loss_config = dict(
            align_loss_weight = codebook_cfg.align_loss_config.align_loss_weight
        )
    else:
        align_loss_config = dict()

    transformer_codebook_config = dict(
        use_codebook = True,
        num_vars = codebook_cfg.num_vars,
        temp_start = codebook_cfg.temp_start,
        temp_stop = codebook_cfg.temp_stop,
        temp_decay = codebook_cfg.temp_decay,
        groups = codebook_cfg.groups,
        combine_groups = codebook_cfg.combine_groups,
        weight_proj_depth = codebook_cfg.weight_proj_depth,
        weight_proj_factor = codebook_cfg.weight_proj_factor,

        use_perplex_loss = use_perplex_loss,
        perplex_loss_config = perplex_loss_config,

        use_align_loss = use_align_loss,
        align_loss_config = align_loss_config
    )

    cfm_codebook_config = dict(
        use_codebook = True,
        use_align_loss = use_align_loss
    )

    return transformer_codebook_config, cfm_codebook_config
