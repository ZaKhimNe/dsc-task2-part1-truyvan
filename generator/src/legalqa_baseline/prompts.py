from __future__ import annotations

from .fewshot import question_type


INITIAL_SYSTEM_PROMPT = """Bạn là hệ thống sinh câu trả lời pháp luật tiếng Việt.
Chỉ sử dụng căn cứ được cung cấp. Không tự thêm tên văn bản, Điều, Khoản,
con số, điều kiện hoặc ngoại lệ không có trong câu hỏi và căn cứ.
Bạn chỉ sinh Khối 1 và Khối 3; chương trình sẽ chép nguyên văn Khối 2.
Bắt buộc trả về đủ cả thẻ <LEAD> và <CONCLUSION>."""


TARGETED_RETRY_SYSTEM_PROMPT = """Bạn là hệ thống sinh câu trả lời pháp luật tiếng Việt.
Chỉ sử dụng căn cứ được cung cấp. Không tự thêm tên văn bản, Điều, Khoản,
con số, điều kiện hoặc ngoại lệ không có trong câu hỏi và căn cứ.
Chương trình sẽ chép nguyên văn Khối 2. Chỉ sinh đúng khối và định dạng được
yêu cầu trong nhiệm vụ, không thêm giải thích ngoài thẻ kết quả."""


V8_SFT_SYSTEM_PROMPT = """Bạn là hệ thống sinh câu trả lời pháp luật tiếng Việt.
Chỉ sử dụng căn cứ được cung cấp. Không tự thêm tên văn bản, Điều, Khoản, con số,
điều kiện hoặc ngoại lệ không có trong câu hỏi và căn cứ.
Bạn chỉ sinh Khối 1 và Khối 3; chương trình sẽ chép nguyên văn Khối 2.
Bắt buộc trả về đủ cả thẻ <LEAD> và <CONCLUSION>."""


def _format_metadata(metadata: dict[str, list[str]]) -> str:
    labels = (
        ("Tên văn bản", "named_documents"),
        ("Số hiệu văn bản", "document_numbers"),
        ("Điều", "articles"),
        ("Khoản", "clauses"),
        ("Điểm", "points"),
    )
    lines = []
    for label, key in labels:
        values = metadata.get(key) or []
        if values:
            lines.append(f"{label}: {', '.join(values)}")
    return "\n".join(lines) if lines else "Không trích xuất được metadata căn cứ."


def build_initial_user_prompt(
    question: str,
    context: str,
    citation_metadata: dict[str, list[str]],
    *,
    legacy_retry: bool = False,
    fewshot_examples: list[dict] | None = None,
) -> str:
    """Prompt nhiệm vụ V4 nguyên văn, có thể đặt ví dụ train ở phía trước."""

    retry_note = """
YÊU CẦU SỬA ĐỊNH DẠNG:
Lần sinh trước không có đủ hai thẻ. Hãy sinh lại từ đầu và bắt buộc có cả
<LEAD>...</LEAD> lẫn <CONCLUSION>...</CONCLUSION>.
""" if legacy_retry else ""
    example_sections = []
    for index, example in enumerate(fewshot_examples or [], start=1):
        example_sections.append(
            f"""VÍ DỤ {index}:
CÂU HỎI VÍ DỤ:
{example["question"]}

METADATA CĂN CỨ VÍ DỤ:
{_format_metadata(example["oracle_context"]["citation_metadata"])}

CĂN CỨ PHÁP LUẬT VÍ DỤ:
{example["oracle_context"]["text"]}

KẾT QUẢ VÍ DỤ:
<LEAD>{example["target"]["lead"]}</LEAD>
<CONCLUSION>{example["target"]["conclusion"]}</CONCLUSION>"""
        )
    examples = ""
    if example_sections:
        examples = (
            "CÁC VÍ DỤ TRAIN CHỈ ĐỂ HỌC CÁCH TRẢ LỜI:\n"
            "Không dùng sự kiện, căn cứ hay kết luận của ví dụ để trả lời câu hỏi mới.\n\n"
            + "\n\n".join(example_sections)
            + "\n\n--- HẾT VÍ DỤ; BẮT ĐẦU CÂU HỎI MỚI ---\n\n"
        )
    return f"""{examples}CÂU HỎI:
{question}

METADATA CĂN CỨ:
{_format_metadata(citation_metadata)}

CĂN CỨ PHÁP LUẬT:
{context}

NHIỆM VỤ:
1. Viết Khối 1: câu dẫn tự nhiên để giới thiệu căn cứ. Giữ chính xác mọi tên
   văn bản và Điều/Khoản nếu chúng xuất hiện trong căn cứ. Chỉ viết một câu,
   tối đa 60 từ và kết thúc bằng dấu hai chấm.
2. Viết Khối 3: kết luận trực tiếp cho câu hỏi, tối đa 120 từ và không chép lại
   toàn bộ căn cứ.
3. Không sinh Khối 2.
4. Không được bỏ trống hoặc bỏ sót một trong hai thẻ.
{retry_note}

Chỉ trả về đúng định dạng sau, không thêm giải thích:
<LEAD>...</LEAD>
<CONCLUSION>...</CONCLUSION>"""


def build_v8_sft_user_prompt(
    question: str,
    context: str,
    citation_metadata: dict[str, list[str]],
    *,
    target: str = "both",
) -> str:
    """Prompt tối thiểu dùng thống nhất khi huấn luyện và suy luận V8."""

    answer_requirements = {
        "subject": (
            "Nêu đầy đủ tất cả chủ thể/đối tượng phù hợp và điều kiện hoặc ngoại lệ "
            "đi kèm nếu căn cứ có quy định."
        ),
        "number_or_time": (
            "Nêu đầy đủ các con số, mức, tỷ lệ hoặc mốc thời gian có liên quan, kể cả "
            "trường hợp và ngoại lệ tương ứng."
        ),
        "condition": (
            "Nêu đầy đủ từng điều kiện, trường hợp và ngoại lệ có liên quan; không chỉ "
            "dẫn chiếu chung chung."
        ),
        "rights_and_duties": (
            "Nêu đầy đủ các quyền, quyền lợi, nghĩa vụ hoặc trách nhiệm mà câu hỏi "
            "yêu cầu; nếu câu hỏi hỏi từ hai phía trở lên thì không được bỏ sót một phía."
        ),
        "procedure_or_list": (
            "Liệt kê đầy đủ từng bước hoặc từng mục liên quan trong căn cứ; không thay "
            "nội dung bằng các cụm như “nêu trên”, “như trên” hoặc “theo quy định trên”."
        ),
        "yes_no": (
            "Trả lời rõ có/không hoặc được/không được, sau đó nêu đầy đủ điều kiện và "
            "ngoại lệ liên quan."
        ),
        "other": (
            "Trả lời trực tiếp và đầy đủ, giữ các điều kiện, trường hợp và ngoại lệ có "
            "liên quan trong căn cứ."
        ),
    }[question_type(question)]

    shared = f"""CÂU HỎI:
{question}

METADATA CĂN CỨ:
{_format_metadata(citation_metadata)}

CĂN CỨ PHÁP LUẬT:
{context}
"""
    if target == "both":
        task = f"""NHIỆM VỤ:
1. Viết Khối 1 là câu dẫn tự nhiên, chính xác để giới thiệu căn cứ pháp luật.
2. Viết Khối 3 là kết luận trực tiếp, đầy đủ cho câu hỏi và không chép lại toàn bộ căn cứ.
3. Không sinh Khối 2.
4. Yêu cầu nội dung Khối 3: {answer_requirements}

Chỉ trả về đúng định dạng sau, không thêm giải thích:
<LEAD>...</LEAD>
<CONCLUSION>...</CONCLUSION>"""
    elif target == "lead":
        task = """NHIỆM VỤ RETRY KHỐI 1:
Chỉ viết câu dẫn tự nhiên, chính xác để giới thiệu căn cứ. Không sinh Khối 2 hoặc Khối 3.

Chỉ trả về:
<LEAD>...</LEAD>"""
    elif target == "conclusion":
        task = f"""NHIỆM VỤ RETRY KHỐI 3:
Chỉ viết kết luận trực tiếp, đầy đủ cho câu hỏi dựa hoàn toàn trên căn cứ. Không sinh Khối 1
hoặc Khối 2 và không chép lại toàn bộ căn cứ.
Yêu cầu nội dung: {answer_requirements}

Chỉ trả về:
<CONCLUSION>...</CONCLUSION>"""
    else:
        raise ValueError(f"Target V8 không hợp lệ: {target}")
    return f"{shared}\n{task}"


def build_v8_sft_completion(row: dict) -> str:
    """Target SFT chỉ chứa hai khối mà model phải tự sinh."""

    lead = str(row["target"]["lead"]).strip()
    conclusion = str(row["target"]["conclusion"]).strip()
    if not lead or not conclusion:
        raise ValueError(f"Mẫu {row.get('id')} thiếu lead hoặc conclusion")
    return f"<LEAD>{lead}</LEAD>\n<CONCLUSION>{conclusion}</CONCLUSION>"


def build_targeted_retry_prompt(
    question: str,
    context: str,
    citation_metadata: dict[str, list[str]],
    *,
    target: str,
) -> str:
    if target == "lead":
        task = """NHIỆM VỤ RETRY KHỐI 1:
Lần sinh trước bị thiếu Khối 1. Chỉ viết một câu dẫn tự nhiên giới thiệu căn cứ,
tối đa 60 từ, giữ chính xác tên văn bản và Điều/Khoản nếu có, kết thúc bằng dấu
hai chấm. Không sinh Khối 2 hoặc Khối 3.

Chỉ trả về:
<LEAD>...</LEAD>"""
    elif target == "conclusion":
        task = """NHIỆM VỤ RETRY KHỐI 3:
Lần sinh trước bị thiếu Khối 3. Chỉ viết kết luận trực tiếp trả lời câu hỏi, tối
đa 120 từ, dựa hoàn toàn trên căn cứ và không chép lại toàn bộ căn cứ. Không sinh
Khối 1 hoặc Khối 2.

Chỉ trả về:
<CONCLUSION>...</CONCLUSION>"""
    else:
        raise ValueError(f"Target retry không hợp lệ: {target}")

    return f"""CÂU HỎI:
{question}

METADATA CĂN CỨ:
{_format_metadata(citation_metadata)}

CĂN CỨ PHÁP LUẬT:
{context}

{task}"""


def build_user_prompt(
    question: str,
    context: str,
    citation_metadata: dict[str, list[str]],
    *,
    target: str = "both",
) -> str:
    """Compatibility dispatcher used by tests and external callers."""

    if target == "both":
        return build_initial_user_prompt(question, context, citation_metadata)
    return build_targeted_retry_prompt(
        question,
        context,
        citation_metadata,
        target=target,
    )


def build_fallback_lead(metadata: dict[str, list[str]]) -> str:
    """Tạo câu dẫn an toàn khi model vẫn thiếu Khối 1 sau lần retry."""

    ordered_keys = ("points", "clauses", "articles")
    references = [
        value.strip()
        for key in ordered_keys
        for value in (metadata.get(key) or [])
        if value.strip()
    ]
    named_documents = [
        value.strip()
        for value in (metadata.get("named_documents") or [])
        if value.strip()
    ]
    document_numbers = [
        value.strip()
        for value in (metadata.get("document_numbers") or [])
        if value.strip()
    ]
    documents = named_documents or document_numbers
    citation = " ".join(references + documents).strip()
    if citation:
        return f"Căn cứ theo {citation}, quy định liên quan như sau:"
    return "Căn cứ nội dung pháp luật được cung cấp, quy định liên quan như sau:"
