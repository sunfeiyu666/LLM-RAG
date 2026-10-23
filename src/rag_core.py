"""基础链路使用的文本块结构及原有向量化预处理。"""
import re
from dataclasses import dataclass

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    context: str


SYNONYMS = {
    "硬盘空间": "存储目录配额",
    "空间满": "超过配额",
    "夜里": "晚上",
    "单人": "一人",
    "带出去": "外带导出",
    "密码": "口令密钥",
    "钥匙": "密钥",
    "离开团队": "离组",
    "坏了": "故障损坏",
    "器材出问题": "设备故障",
    "开盖检修": "自行拆卸",
    "摄像": "拍照录音",
    "录像": "拍照录音",
    "外来人员": "访客",
    "借东西": "设备借用",
    "报账": "报销",
    "机房": "实验室",
}


SYNONYMS.update({
    "保研": "保研 推荐 免试 推免", "软工": "软件工程", "网安": "网络空间安全",
    "数媒": "数字媒体技术", "挂科": "不及格", "绩点": "绩点 GPA",
    "退课": "退选 退课", "补选": "补选 选课", "先修": "先修 先修课",
})


def semantic_normalize(text: str) -> str:
    result = text
    for src, dst in SYNONYMS.items():
        result = result.replace(src, f"{src} {dst}")
    return result


def dense_prepare(text: str) -> str:
    """保留原有同义表达扩展和字母数字预处理，维持TF-IDF/LSA向量化行为。"""
    return re.sub(r"[A-Za-z0-9_.:-]+", " ", semantic_normalize(text))
