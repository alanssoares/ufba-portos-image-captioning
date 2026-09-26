from src.config import load_config, parse_override, resolve_path, stage_config


def test_defaults_carregam():
    cfg = load_config()
    assert cfg.model.llm_id == "menezesbruno/manaca-1b-base"
    assert cfg.labeling.model == "claude-opus-5-5"
    assert cfg.variants == ["base", "pretrain", "finetune", "lora", "qlora", "gold"]


def test_override_tipado():
    assert parse_override("training.lora.r=32") == (["training", "lora", "r"], 32)
    assert parse_override("evaluation.bertscore=false") == (["evaluation", "bertscore"], False)
    assert parse_override("training.lora.lr=1e-4")[1] == 1e-4
    assert parse_override("a.b=[x, y]")[1] == ["x", "y"]


def test_perfil_e_set_se_sobrepoem():
    cfg = load_config(["configs/perfis/colab_t4.yaml"], ["training.finetune.unfreeze_last_n_layers=4"])
    ft = stage_config(cfg, "finetune")
    assert ft.llm_mode == "partial"
    assert ft.unfreeze_last_n_layers == 4
    assert ft.epochs == 3            # vem do default
    assert ft.init_from == "pretrain"


def test_stage_herda_common():
    cfg = load_config(overrides=["training.common.batch_size=7"])
    for estagio in ("pretrain", "finetune", "lora", "qlora"):
        assert stage_config(cfg, estagio).batch_size == 7
    q = stage_config(cfg, "qlora")
    assert q.quantize_4bit is True and q.r == 16 and "q_proj" in q.target_modules


def test_flag_init_from():
    cfg = load_config(overrides=["training.lora.init_from=base"])
    assert stage_config(cfg, "lora").init_from == "base"
    assert stage_config(cfg, "qlora").init_from == "pretrain"


def test_caminhos_relativos_a_root(tmp_path):
    cfg = load_config(overrides=[f"paths.root={tmp_path.as_posix()}"])
    assert resolve_path(cfg, "labels_jsonl") == tmp_path / "data" / "labels.jsonl"


def test_todos_os_perfis_carregam():
    for perfil in ("smoke", "colab_t4", "colab_l4", "colab_a100", "local_4gb"):
        cfg = load_config([f"configs/perfis/{perfil}.yaml"])
        for estagio in ("base", "pretrain", "finetune", "lora", "qlora"):
            stage_config(cfg, estagio)
