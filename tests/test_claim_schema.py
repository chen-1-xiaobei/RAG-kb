"""入参 schema 与安全类型转换的测试。

背景：模型给的参数是"脏"的——它会把 `"450"`、`"450元"`、`true`、`NaN`、`"FP-1, FP-2"`
混着给你。裸 `float()` / `int()` 挡不住这些：`float(True) == 1.0` 会让"金额=true"
静默变成 1 块钱，比报错难查得多。

所以这里断言两件事：
1. **能安全归一化的都归一化**（字符串数字、货币符号、别名、逗号分隔的发票号）
2. **归不了或危险的明确报错**（布尔、NaN/Inf、负数、非线性晚数、未知字段）
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from claims import (  # noqa: E402
    MAX_INVOICES,
    Claim,
    ClaimValidationError,
    parse_claim,
)


# ---------------------------------------------------------------------------
# 归一化：该接受的要接受
# ---------------------------------------------------------------------------

def test_string_amount_with_currency_noise():
    assert parse_claim(category="住宿", amount="450元").amount == 450.0
    assert parse_claim(category="住宿", amount=" ¥1,234.5 ").amount == 1234.5


def test_amount_rounds_to_two_decimals():
    assert parse_claim(category="住宿", amount=450.678).amount == 450.68


def test_category_aliases_are_normalized():
    assert parse_claim(category="住宿费", amount=1).category == "住宿"
    assert parse_claim(category="打车", amount=1).category == "市内交通"
    assert parse_claim(category="机票", amount=1).category == "飞机"


def test_unknown_category_is_kept_for_policy_to_judge():
    """类型层不认识的类别原样保留——由 evaluate_policy 判成"未知类别"。

    把业务枚举塞进 schema 会让"政策拒绝"变成"422/异常"，两者的用户话术不同。
    """
    assert parse_claim(category="赌博", amount=10).category == "赌博"


def test_city_aliases_map_city_names_to_tier():
    for city in ("北京", "上海", "广州", "深圳", "一线城市"):
        assert parse_claim(category="住宿", amount=1, city_tier=city).city_tier == "一线"
    assert parse_claim(category="住宿", amount=1, city_tier="二线").city_tier == "其他"


def test_nights_accepts_integral_forms():
    assert parse_claim(category="住宿", amount=1, nights="1").nights == 1
    assert parse_claim(category="住宿", amount=1, nights=2.0).nights == 2
    assert parse_claim(category="住宿", amount=1, nights="").nights == 0


def test_invoice_ids_accepts_comma_string_and_dedupes():
    claim = parse_claim(category="住宿", amount=1, invoice_ids="FP-1, FP-2，FP-1、FP-3")
    assert claim.invoice_ids == ("FP-1", "FP-2", "FP-3")


def test_invoice_ids_none_becomes_empty_tuple():
    assert parse_claim(category="餐饮补贴", amount=100, invoice_ids=None).invoice_ids == ()


# ---------------------------------------------------------------------------
# 拒绝：危险或无法解析的必须报错
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("amount", [True, False])
def test_boolean_amount_is_rejected(amount):
    """`float(True) == 1.0` 是 Python 的经典坑：金额=true 不该变成 1 块钱。"""
    with pytest.raises(ClaimValidationError) as excinfo:
        parse_claim(category="住宿", amount=amount)
    assert "布尔" in str(excinfo.value)


@pytest.mark.parametrize("amount", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_amount_is_rejected(amount):
    with pytest.raises(ClaimValidationError) as excinfo:
        parse_claim(category="住宿", amount=amount)
    assert "有限" in str(excinfo.value)


@pytest.mark.parametrize("amount", [0, -1, -0.01])
def test_non_positive_amount_is_rejected(amount):
    with pytest.raises(ClaimValidationError):
        parse_claim(category="住宿", amount=amount)


def test_unparsable_amount_is_rejected_with_original_value():
    with pytest.raises(ClaimValidationError) as excinfo:
        parse_claim(category="住宿", amount="大约五百")
    assert "大约五百" in str(excinfo.value)


def test_boolean_nights_is_rejected():
    with pytest.raises(ClaimValidationError):
        parse_claim(category="住宿", amount=1, nights=True)


def test_fractional_nights_is_rejected():
    with pytest.raises(ClaimValidationError):
        parse_claim(category="住宿", amount=1, nights=1.5)


def test_unknown_field_is_rejected():
    with pytest.raises(ClaimValidationError) as excinfo:
        parse_claim(category="住宿", amount=1, approver="ceo")
    assert "approver" in str(excinfo.value)


def test_too_many_invoices_is_rejected():
    with pytest.raises(ClaimValidationError):
        parse_claim(category="住宿", amount=1,
                    invoice_ids=[f"FP-{index}" for index in range(MAX_INVOICES + 1)])


def test_claim_is_frozen():
    claim = parse_claim(category="住宿", amount=1)
    with pytest.raises(Exception):
        claim.amount = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 与既有代码的兼容
# ---------------------------------------------------------------------------

def test_keyword_construction_still_works():
    """老代码与研究脚本都用关键字构造，不能被 schema 改造打断。"""
    claim = Claim(category="住宿", amount=500.0, nights=1, city_tier="一线",
                  invoice_ids=("i",))
    assert claim.as_dict() == {
        "category": "住宿", "amount": 500.0, "nights": 1,
        "city_tier": "一线", "invoice_ids": ["i"],
    }


def test_as_dict_roundtrip_through_parse():
    original = parse_claim(category="高铁", amount="123.4", invoice_ids="FP-9")
    again = parse_claim(**original.as_dict())
    assert again == original
