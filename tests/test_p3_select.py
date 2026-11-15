import json

import pytest

from p3 import select as sel


def arm(logical, bits, acc, diverged=False, seeds=5):
    return [{"logical": logical, "bits": dict(bits), "seed": s,
             "fp16_val_acc": acc + 0.0002 * (s - seeds // 2),
             "diverged": diverged, "tok_per_s": 100.0 + s}
            for s in range(1, seeds + 1)]


def task(task_id, cls, ref, arms):
    trials = []
    for a in arms:
        trials.extend(a)
    return {"id": task_id, "mechanism_class": cls, "fp16_val_acc": ref,
            "seeds": list(range(1, 6)), "trials": trials}


def pool(tasks, **extra):
    doc = {"kind": "hgk_p3_pool", "tasks": tasks}
    doc.update(extra)
    return doc


def test_best_eligible_config_wins_and_ties_break_on_bits():
    t = task("t1", "rmsnorm", 0.600, [
        arm(16, {"w": 16}, 0.5998),
        arm(4, {"w": 7}, 0.5998),
        arm(3, {"w": 6}, 0.5998),      # identical mean, fewest bits -> should win
        arm(2, {"w": 2}, 0.580),       # R2 reject
    ])
    row = sel.evaluate_task(t, 0.600, {})
    assert row["selected"] is True
    assert row["best"]["logical"] == 3
    assert any("mean 0.58" in r for r in row["reasons"])


def test_diverged_arm_is_excluded_even_if_more_accurate():
    t = task("t1", "rmsnorm", 0.600, [
        arm(16, {"w": 16}, 0.600),
        arm(3, {"w": 6}, 0.601, diverged=True),
    ])
    row = sel.evaluate_task(t, 0.600, {})
    assert row["best"]["logical"] == 16
    assert any("R1" in r and "diverged" in r for r in row["reasons"])


def test_diverged_fp16_control_excludes_the_whole_task():
    t = task("t1", "rmsnorm", 0.600, [
        arm(16, {"w": 16}, 0.600, diverged=True),
        arm(3, {"w": 6}, 0.600),
    ])
    row = sel.evaluate_task(t, 0.600, {})
    assert row["selected"] is False
    assert row["best"] is None
    assert any("FP16 control arm diverged" in r for r in row["reasons"])


def test_pool_wide_floor_rejects_a_low_task():
    good = task("good", "softmax", 0.700, [arm(16, {"w": 16}, 0.700),
                                           arm(3, {"w": 6}, 0.6999)])
    # 0.640 - 0.6999 is far beyond the 0.0035 pool slack
    bad = task("bad", "softmax", 0.640, [arm(16, {"w": 16}, 0.640),
                                         arm(3, {"w": 6}, 0.6400)])
    doc = sel.select(pool([good, bad]))
    rows = {r["task_id"]: r for r in doc["tasks"]}
    assert rows["good"]["selected"] is True
    assert rows["bad"]["selected"] is False
    assert any(r.startswith("R3:") for r in rows["bad"]["reasons"])


def test_caveat_blocks_a_task_that_only_matches_a_broken_reference():
    t = task("cav", "gemv", 0.500, [arm(16, {"w": 16}, 0.500),
                                    arm(3, {"w": 6}, 0.4998)])
    doc = sel.select(pool([t], fp16_acc_caveat={"cav": 0.4990}))
    assert doc["tasks"][0]["selected"] is False
    assert any("R3.b" in r for r in doc["tasks"][0]["reasons"])


def test_family_needs_three_selected_tasks():
    two = [task(f"t{i}", "rmsnorm", 0.600, [arm(16, {"w": 16}, 0.600),
                                            arm(3, {"w": 6}, 0.5999)])
           for i in range(2)]
    doc = sel.select(pool(two))
    assert doc["family"] is None
    assert doc["selected_mechanisms"] == ["rmsnorm"]

    three = two + [task("t2b", "rmsnorm", 0.600,
                        [arm(16, {"w": 16}, 0.600), arm(3, {"w": 6}, 0.5999)])]
    doc = sel.select(pool(three))
    assert doc["family"]["mechanism_class"] == "rmsnorm"
    assert doc["family"]["task_count"] == 3


def test_joint_requires_exact_measurement_and_labels_the_planning_sum():
    tasks = [task(f"t{i}", "rmsnorm", 0.600, [arm(16, {"w": 16}, 0.600),
                                              arm(3, {"w": 6}, 0.5999)])
             for i in range(3)]
    doc = sel.select(pool(tasks, per_class_impact_pct={"rmsnorm": 1.8}))
    assert doc["joint"]["status"] == "requires_joint_measurement"
    assert doc["joint"]["value"] is None
    assert doc["joint"]["planning_sum_pct"] == pytest.approx(1.8)

    measured = sel.select(pool(tasks, per_class_impact_pct={"rmsnorm": 1.8},
                               joint={"mechanisms": ["rmsnorm"], "value": 1.4,
                                      "source": "evidence/p3/joint.json"}))
    assert measured["joint"]["status"] == "measured"
    assert measured["joint"]["value"] == pytest.approx(1.4)

    wrong_set = sel.select(pool(tasks, joint={"mechanisms": ["rmsnorm", "gemv"],
                                              "value": 2.0}))
    assert wrong_set["joint"]["status"] == "requires_joint_measurement"


def test_selection_is_deterministic():
    tasks = [task(f"t{i}", "softmax", 0.600, [arm(16, {"w": 16}, 0.600),
                                              arm(3, {"w": 6}, 0.5999)])
             for i in range(3)]
    doc = pool(tasks)
    assert json.dumps(sel.select(doc), sort_keys=True) == \
        json.dumps(sel.select(doc), sort_keys=True)


def test_policy_hash_is_pinned():
    assert sel.verify_policy() == "46eb46c1945f8b09b3ba579c2d485a6e042265669672207a0dd9587f8e025fa8"


def test_policy_edit_is_detected(tmp_path):
    policy = tmp_path / "policy.md"
    frozen = tmp_path / "policy.sha256"
    policy.write_text("frozen text\n")
    frozen.write_text(sel.sha256_bytes(policy.read_bytes()) + "\n")
    assert sel.verify_policy(policy, frozen) == sel.sha256_file(policy)

    policy.write_text("edited after freeze\n")
    with pytest.raises(SystemExit, match="edited"):
        sel.verify_policy(policy, frozen)


def test_empty_pool_is_refused():
    with pytest.raises(SystemExit, match="no tasks"):
        sel.select({"tasks": []})


def test_main_refuses_a_missing_pool(tmp_path):
    with pytest.raises(SystemExit, match="missing pool"):
        sel.main(["--pool", str(tmp_path / "nope.json"),
                  "--out", str(tmp_path / "out.json")])


def test_config_id_and_total_bits():
    assert sel.config_id(3, {"grad_hi": 3, "grad_lo": 2}) == "L3:grad_hi=3,grad_lo=2"
    assert sel.total_bits({"a": 3, "b": 2}) == 5
