import json
import os
import time

from loguru import logger as eval_logger

from lmms_eval.llm_judge import Request, ServerConfig, get_server

API_TYPE = os.getenv("API_TYPE", "openai")
MODEL_VERSION = os.getenv("MODEL_VERSION", "gpt-4o-2024-11-20")

RETRY_TIMES = 16

server_config = ServerConfig(model_name=MODEL_VERSION, temperature=1.0, max_tokens=256)
server = get_server(server_name=API_TYPE, config=server_config)

# Official SimpleVQA judge prompt from https://github.com/SimpleVQA/SimpleVQA
SIMPLEVQA_JUDGE_PROMPT = """请根据给定问题、标准答案和模型预测的答案来评估模型的回答是否正确。您的任务是将结果评定为：【正确】、【错误】或【未尝试】。
首先，我们将列出每个评定类别的示例，然后请您对新问题的预测答案进行评定。
## 以下是【正确】的答复示例：
```
问题：图中人物（贝拉克·奥巴马）的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：Malia Obama and Sasha Obama
模型预测2：玛丽亚和萨沙
模型预测3：大多数人会说是玛丽亚和萨莎，但我不确定，需要再确认
模型预测4：巴拉克·奥巴马有两个女儿，她们分别是玛丽亚·安和娜塔莎·玛丽安，但通常称作玛丽亚·奥巴马和萨莎·奥巴马。玛丽亚出生于1998年7月4日，萨莎出生于2001年6月10日。
```
这些答复均为【正确】，因为：
- 完整地包含了标准答案中的重要信息。
- 不包含任何与标准答案矛盾的信息。
- 只关注语义内容，中英文，大小写、标点、语法和顺序不重要。
- 答复中出现模糊语句或猜测是可以接受的，前提是包含了标准答案且不含有不正确信息或矛盾。

## 以下是【错误】的答复示例：
```
问题：图中人物（贝拉克·奥巴马）的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：玛丽亚
模型预测2：玛丽亚、萨莎和苏珊
模型预测3：巴拉克·奥巴马没有孩子
模型预测4：我认为是玛丽亚和萨莎。或者是玛丽亚和杰基。或者是乔伊和玛丽亚。
模型预测5：虽然我不知道他们的确切名字，但能说出巴拉克·奥巴马有三个孩子。
模型预测6：你可能是想说贝茜和奥利维亚。不过您应通过最新的参考资料确认详细信息。那是正确的答案吗？
```
这些答复均为【错误】，因为：
- 答复中包含与标准答案矛盾的事实陈述。即使在陈述中略带保留（例如："可能是"，"虽然我不确定，但我认为"），也视为错误。

## 以下是【未尝试】的答复示例：
```
问题：图中人物（贝拉克·奥巴马）的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：我不知道。
模型预测2：我需要更多关于您所指奥巴马的上下文。
模型预测3：不查阅网络我无法回答这个问题，不过我知道巴拉克·奥巴马有两个孩子。
模型预测4：巴拉克·奥巴马有两个孩子。我知道其中一个叫玛丽亚，但我不确定另一个的名字。
模型预测5：我无法识别图中的人物。
模型预测6：N/A。
```
这些答复均为【未尝试】，因为：
- 没有包含标准答案中的重要信息。
- 回复中没有与标准答案矛盾的陈述。
只返回字母"A"、"B"或"C"，无须添加其他文本。

另外注意以下几点：
- 对于标准答案为数字的问题，预测答案应和标准答案一致。例如，考虑问题"金山铁路黄浦江特大桥的全长是多少米？"，标准答案为"3518.17"：
- 预测答案"3518"、"3518.1"、"3518.17"均为【正确】。
- 预测答案"3520"和"3600"均为【错误】。
- 预测答案"大约3500米"和"超过3000米"被视为【未尝试】，因为它们既不确认也不与标准答案矛盾。
- 如果标准答案包含比问题更多的信息，预测答案只需包含问题中提到的信息。
- 例如，考虑问题"菱镁矿的主要化学成分是什么？",标准答案为"碳酸镁（MgCO3）"。"碳酸镁"或"MgCO3"均视为【正确】答案。
- 如果从问题中明显可以推断出预测答案省略的信息，那么算作【正确】。
- 例如，问题"巴鲁米尼的努拉吉遗迹在1997年被联合国教科文组织列为世界文化遗产，那么这遗址在哪个地区？"标准答案为"意大利撒丁岛"，预测答案"撒丁岛"被视为【正确】。
- 如果能明显看出名字翻译版本不同但是是同一个人也认为【正确】。
- 例如，如果标准答案是"Robinson"，那么回答鲁滨逊或者鲁滨孙均【正确】。
- 预测答案和标准答案对应的是同一事物，但是称呼不同，如"天坛"和"祈年殿"，那么算作【正确】

## 下面是一个新的问题示例。对每一个预测答案，请只回复"正确"、"错误"、"未尝试"之一，不要道歉或纠正自己的错误，只需要评估该回答。
```
问题: {question}
正确答案: {answer}
预测答案: {candidates}
```

请严格按照以下格式回复，以JSON格式返回一个字典,而且字典第一层key不要替换为具体答案。不要返回其他任何内容。
[回复格式]:
```json
{{
    "预测答案0": "整体结论，只返回'正确','错误','未尝试'中的一个，无须添加其他文本"
}}
```"""

JUDGE_LABEL_MAP = {"正确": "correct", "错误": "incorrect", "未尝试": "not_attempted"}


def process_en(dataset):
    return dataset.filter(lambda x: x["language"] == "EN")


def process_cn(dataset):
    return dataset.filter(lambda x: x["language"] == "CN")


def simplevqa_doc_to_visual(doc):
    if doc["image"] is None:
        return []
    return [doc["image"].convert("RGB")]


def simplevqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    if lmms_eval_specific_kwargs is None:
        return question
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return f"{pre_prompt}{question}{post_prompt}"


def _call_judge(prompt: str, patience: int = RETRY_TIMES) -> tuple[str, str]:
    """Call the LLM judge with retry logic."""
    custom_config = ServerConfig(
        model_name=MODEL_VERSION, temperature=1.0, max_tokens=256
    )

    while patience > 0:
        patience -= 1
        try:
            request = Request(
                messages=[{"role": "user", "content": prompt}], config=custom_config
            )
            response = server.evaluate(request)
            content = response.content.strip() if response.content else ""
            if content:
                return content, response.model_used
        except Exception as e:
            eval_logger.error(f"Error: {e}")
            if "Rate limit" in str(e):
                time.sleep(5)
            eval_logger.info(f"Retrying...Patience left: {patience}")

    return "", ""


def _parse_judge_response(raw: str) -> str:
    """Parse the judge JSON response and return one of: correct, incorrect, not_attempted."""
    raw = raw.replace("```json", "").replace("```python", "").replace("```", "").strip()
    if raw and raw[-1] != "}":
        raw += "}"

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return "not_attempted"

    # The response dict has a single key like "预测答案0"
    for value in parsed.values():
        label = value if isinstance(value, str) else value.get("conclusion", "")
        label = label.replace("**", "").strip()
        if label in JUDGE_LABEL_MAP:
            return JUDGE_LABEL_MAP[label]
        # Fallback: if the label is too long or unrecognized, treat as not_attempted
        return "not_attempted"

    return "not_attempted"


def simplevqa_process_results(doc, results):
    pred = results[0]
    question = doc["question"]
    answer = doc["answer"]

    candidates = f"\n[预测答案0]：{pred}"
    prompt = SIMPLEVQA_JUDGE_PROMPT.format(
        question=question, answer=answer, candidates=candidates
    )

    raw_response, model_name = _call_judge(prompt)
    judgment = _parse_judge_response(raw_response) if raw_response else "not_attempted"

    if not raw_response:
        eval_logger.warning(f"data_id={doc['data_id']} failed to get a judge response.")

    result_dict = {
        "data_id": doc["data_id"],
        "question": question,
        "gt_answer": answer,
        "pred_answer": pred,
        "judgment": judgment,
        "raw_judge_response": raw_response,
        "eval_model": model_name,
    }

    return {
        "simplevqa_is_correct": result_dict,
        "simplevqa_accuracy_given_attempted": result_dict,
        "simplevqa_f1": result_dict,
    }


def simplevqa_aggregate_is_correct(results: list[dict]) -> float:
    """Proportion of correct answers (official 'is_correct' metric)."""
    if not results:
        return 0.0
    n_correct = sum(1 for r in results if r["judgment"] == "correct")
    score = n_correct / len(results)
    eval_logger.info(f"SimpleVQA is_correct: {score:.4f} ({n_correct}/{len(results)})")
    return score


def simplevqa_aggregate_accuracy_given_attempted(results: list[dict]) -> float:
    """Accuracy among attempted answers (correct / (correct + incorrect))."""
    if not results:
        return 0.0
    n_correct = sum(1 for r in results if r["judgment"] == "correct")
    n_incorrect = sum(1 for r in results if r["judgment"] == "incorrect")
    n_attempted = n_correct + n_incorrect
    if n_attempted == 0:
        return 0.0
    score = n_correct / n_attempted
    eval_logger.info(
        f"SimpleVQA accuracy_given_attempted: {score:.4f} ({n_correct}/{n_attempted})"
    )
    return score


def simplevqa_aggregate_f1(results: list[dict]) -> float:
    """F1 score: harmonic mean of is_correct and accuracy_given_attempted."""
    if not results:
        return 0.0
    n_correct = sum(1 for r in results if r["judgment"] == "correct")
    n_incorrect = sum(1 for r in results if r["judgment"] == "incorrect")
    n_attempted = n_correct + n_incorrect
    is_correct = n_correct / len(results)
    acc_attempted = n_correct / n_attempted if n_attempted > 0 else 0.0
    if (acc_attempted + is_correct) == 0:
        return 0.0
    f1 = 2 * acc_attempted * is_correct / (acc_attempted + is_correct)
    eval_logger.info(f"SimpleVQA f1: {f1:.4f}")
    return f1
