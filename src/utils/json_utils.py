# src/utils/hashing.py
# -*- coding: utf-8 -*-

import json
import re

def extract_json_string(text: str) -> str | None:
    """
    从可能包含前后无关文本的字符串中提取第一个有效的 JSON 对象或数组。

    Args:
        text: 包含潜在 JSON 的输入字符串。

    Returns:
        如果找到有效的 JSON，则返回该 JSON 字符串，否则返回 None。
    """
    if not isinstance(text, str):
        return None

    # 1. 尝试使用正则表达式查找被 ```json ... ``` 或 ``` ... ``` 包裹的内容
    #    使用 re.DOTALL 使 '.' 可以匹配换行符
    match = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL)
    if match:
        potential_json = match.group(1)
        try:
            json.loads(potential_json)
            return potential_json
        except json.JSONDecodeError:
            # 如果正则匹配到的内容不是有效的 JSON，继续尝试下面的方法
            pass

    # 2. 如果没有找到代码块或代码块内容无效，则查找第一个 '{' 或 '['
    #    以及最后一个 '}' 或 ']'
    first_bracket = text.find('[')
    first_brace = text.find('{')

    if first_brace == -1 and first_bracket == -1:
        # 没有找到 JSON 的起始符号
        return None

    if first_brace == -1:
        start_index = first_bracket
        start_char = '['
    elif first_bracket == -1:
        start_index = first_brace
        start_char = '{'
    else:
        # 两者都找到了，取更早出现的那个
        start_index = min(first_brace, first_bracket)
        start_char = '{' if start_index == first_brace else '['

    # 确定对应的结束符号
    end_char = '}' if start_char == '{' else ']'

    # 寻找最后一个对应的结束符号
    # 从 start_index 开始查找，以避免找到 JSON 之前的结束符号
    last_end_char_index = text.rfind(end_char, start_index)

    if last_end_char_index == -1 or last_end_char_index < start_index:
        # 没有找到有效的结束符号
        return None

    # 尝试从第一个开始符号到最后一个结束符号提取子串
    potential_json = text[start_index : last_end_char_index + 1]

    try:
        json.loads(potential_json)
        return potential_json
    except json.JSONDecodeError:
        # 如果上面的尝试失败，可能是因为最后一个结束符不匹配第一个开始符
        # (例如，文本是 "{...} [...]")。
        # 我们可以尝试更复杂的逻辑，比如匹配括号层级，
        # 但一个更简单（虽然不完美）的方法是尝试从第一个开始符
        # 到 *任何* 最后一个结束符（'}' 或 ']'）
        last_brace = text.rfind('}', start_index)
        last_bracket = text.rfind(']', start_index)
        last_any_end_index = max(last_brace, last_bracket)

        if last_any_end_index > start_index:
             potential_json_alt = text[start_index : last_any_end_index + 1]
             try:
                 json.loads(potential_json_alt)
                 return potential_json_alt
             except json.JSONDecodeError:
                 # 仍然失败，放弃
                 pass

    # 3. 如果上述所有尝试都失败了
    return None

# --- 示例 ---
text1 = "这是 LLM 的一些说明文字。\n```json\n{\n  \"name\": \"示例 JSON\",\n  \"version\": 1.0,\n  \"enabled\": true\n}\n```\n还有一些结尾文字。"
text2 = "当然，这是您要的 JSON 数据：{\"widget\": {\"debug\": \"on\", \"window\": {\"name\": \"main\", \"width\": 500, \"height\": 500}}}希望对您有帮助！"
text3 = "这是一个列表：[1, 2, \"测试\", {\"key\": \"value\"}]"
text4 = "这里没有 JSON。"
text5 = "无效的 JSON { name: \"John Doe\" } 因为 key 没有引号。" # JSON 标准要求 key 必须是双引号字符串
text6 = "```\n[\"item1\", \"item2\"]\n```"
text7 = "开头 { \"valid\": true } 结尾 ] 这不是有效的json" # 括号不匹配
text8 = "请看: {\"a\": 1} 和 [\"b\", 2]" # 包含多个独立的 JSON

print(f"--- Text 1 ---\nInput:\n{text1}\nExtracted:\n{extract_json_string(text1)}\n")
print(f"--- Text 2 ---\nInput:\n{text2}\nExtracted:\n{extract_json_string(text2)}\n")
print(f"--- Text 3 ---\nInput:\n{text3}\nExtracted:\n{extract_json_string(text3)}\n")
print(f"--- Text 4 ---\nInput:\n{text4}\nExtracted:\n{extract_json_string(text4)}\n")
print(f"--- Text 5 ---\nInput:\n{text5}\nExtracted:\n{extract_json_string(text5)}\n") # 应该提取失败
print(f"--- Text 6 ---\nInput:\n{text6}\nExtracted:\n{extract_json_string(text6)}\n")
print(f"--- Text 7 ---\nInput:\n{text7}\nExtracted:\n{extract_json_string(text7)}\n") # 应该提取失败
print(f"--- Text 8 ---\nInput:\n{text8}\nExtracted:\n{extract_json_string(text8)}\n") # 只会提取第一个