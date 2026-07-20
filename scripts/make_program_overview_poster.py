from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "ironflow_program_overview_poster_ko.png"
FONT_DIR = Path("C:/Windows/Fonts")

W, H = 1600, 2400


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / name), size)


img = Image.new("RGB", (W, H), (244, 247, 251))
d = ImageDraw.Draw(img)

F_TITLE = font("malgunbd.ttf", 58)
F_SUB = font("malgun.ttf", 28)
F_H1 = font("malgunbd.ttf", 38)
F_H2 = font("malgunbd.ttf", 30)
F_BODY = font("malgun.ttf", 25)
F_BODY_B = font("malgunbd.ttf", 25)
F_SMALL = font("malgun.ttf", 21)
F_TINY = font("malgun.ttf", 17)

COL = {
    "ink": (20, 30, 50),
    "muted": (83, 98, 122),
    "line": (194, 206, 222),
    "blue": (39, 96, 225),
    "green": (26, 157, 78),
    "teal": (23, 147, 135),
    "orange": (235, 85, 24),
    "purple": (123, 61, 224),
    "bg2": (255, 255, 255),
    "shadow": (218, 226, 238),
    "dark": (35, 48, 69),
}


def rounded_box(x, y, w, h, r=22, fill=(255, 255, 255), outline=None, width=2, shadow=True):
    if shadow:
        d.rounded_rectangle((x + 8, y + 10, x + w + 8, y + h + 10), r, fill=COL["shadow"])
    d.rounded_rectangle((x, y, x + w, y + h), r, fill=fill, outline=outline or COL["line"], width=width)


def label(x, y, text, color):
    tw = d.textlength(text, font=F_SMALL)
    d.rounded_rectangle((x, y, x + tw + 34, y + 38), 19, fill=color)
    d.text((x + 17, y + 6), text, fill="white", font=F_SMALL)


def wrap_lines(text, font_obj, max_w):
    words = text.split(" ")
    lines = []
    cur = ""
    for word in words:
        test = word if not cur else cur + " " + word
        if d.textlength(test, font=font_obj) <= max_w:
            cur = test
            continue
        if cur:
            lines.append(cur)
        if d.textlength(word, font=font_obj) <= max_w:
            cur = word
            continue
        chunk = ""
        for ch in word:
            if d.textlength(chunk + ch, font=font_obj) <= max_w:
                chunk += ch
            else:
                lines.append(chunk)
                chunk = ch
        cur = chunk
    if cur:
        lines.append(cur)
    return lines


def draw_wrapped(x, y, text, font_obj, fill, max_w, line_h=None):
    if line_h is None:
        line_h = int(font_obj.size * 1.45)
    yy = y
    for line in wrap_lines(text, font_obj, max_w):
        d.text((x, yy), line, fill=fill, font=font_obj)
        yy += line_h
    return yy


def bullet_list(x, y, items, color, max_w, line_h=36):
    yy = y
    for item in items:
        d.ellipse((x, yy + 8, x + 13, yy + 21), fill=color)
        yy = draw_wrapped(x + 28, yy, item, F_BODY, COL["ink"], max_w - 28, line_h)
        yy += 5
    return yy


d.rectangle((0, 0, W, 230), fill=(232, 240, 250))
d.text((70, 54), "IronFlow Vision Experiment Platform", fill=COL["ink"], font=F_TITLE)
d.text(
    (72, 132),
    "모델 후보 선택부터 Vast GPU 학습, 결과 수집과 성능 평가까지 자동화하는 실험 운영 플랫폼",
    fill=(48, 63, 87),
    font=F_SUB,
)
label(72, 178, "훈련 자동화", COL["blue"])
label(232, 178, "원격 GPU 실행", COL["orange"])
label(430, 178, "결과 표준화", COL["green"])
label(610, 178, "팀 단위 비교", COL["purple"])

rounded_box(70, 270, 1460, 170, r=24, fill=COL["bg2"])
d.text((110, 302), "한 줄 요약", fill=COL["blue"], font=F_H2)
draw_wrapped(
    110,
    355,
    "IronFlow는 여러 detector와 classifier 후보를 같은 조건으로 실행하고, 실험 결과를 표준화된 지표와 산출물로 모아 최종 모델 선택을 돕는 도구입니다.",
    F_BODY,
    COL["ink"],
    1360,
    38,
)

d.text((70, 505), "전체 실험 흐름", fill=COL["ink"], font=F_H1)
steps = [
    ("1", "모델 후보 선택", "Detector / Classifier 후보를 GUI에서 선택", "blue"),
    ("2", "실행 조건 설정", "YAML Config와 Run Options를 실험별로 확정", "teal"),
    ("3", "Queue 순차 실행", "최대 5개 실험을 예약하고 Run All로 자동 실행", "purple"),
    ("4", "Vast GPU 학습", "SSH 연결 후 원격 서버에서 학습과 평가 수행", "orange"),
    ("5", "결과 수집/평가", "Metrics, Weight, Preview, Log를 로컬로 수집", "green"),
]
x0, y0 = 90, 565
box_w, box_h, gap = 270, 220, 22
for i, (num, title, body, cname) in enumerate(steps):
    x = x0 + i * (box_w + gap)
    rounded_box(x, y0, box_w, box_h, r=24, fill=COL["bg2"])
    d.rounded_rectangle((x, y0, x + box_w, y0 + 60), 24, fill=COL[cname])
    d.text((x + 24, y0 + 15), f"{num}. {title}", fill="white", font=F_BODY_B)
    draw_wrapped(x + 24, y0 + 92, body, F_BODY, COL["ink"], box_w - 48, 35)
    if i < len(steps) - 1:
        ax = x + box_w + 6
        ay = y0 + 105
        d.line((ax, ay, ax + 20, ay), fill=COL["muted"], width=5)
        d.polygon([(ax + 20, ay - 10), (ax + 20, ay + 10), (ax + 36, ay)], fill=COL["muted"])

section_y = 860
d.text((70, section_y), "동작 메커니즘", fill=COL["ink"], font=F_H1)
rounded_box(70, section_y + 60, 1460, 300, r=24, fill=COL["bg2"])
arch = [
    ("Experiments 탭", "후보 선택"),
    ("Effective Config", "실행 조건 확정"),
    ("Task Adapter", "모델 라이브러리 연결"),
    ("Remote Runner", "GPU 서버 실행"),
    ("Collector", "결과 수집"),
    ("Metrics Builder", "평가 리포트 생성"),
]
ax, ay, aw, ah, ag = 110, section_y + 145, 190, 88, 42
for i, (a, b) in enumerate(arch):
    x = ax + i * (aw + ag)
    d.rounded_rectangle((x, ay, x + aw, ay + ah), 18, fill=(248, 251, 255), outline=COL["line"], width=2)
    d.text((x + 18, ay + 18), a, fill=COL["ink"], font=F_SMALL)
    d.text((x + 18, ay + 50), b, fill=COL["muted"], font=F_SMALL)
    if i < len(arch) - 1:
        lx = x + aw + 8
        ly = ay + 44
        d.line((lx, ly, lx + 22, ly), fill=COL["muted"], width=4)
        d.polygon([(lx + 22, ly - 8), (lx + 22, ly + 8), (lx + 35, ly)], fill=COL["muted"])
draw_wrapped(
    110,
    section_y + 260,
    "GUI에서 선택한 후보와 옵션은 effective config로 고정되고, 각 모델 adapter가 실제 YOLO / TorchVision / Transformer 계열 학습 코드를 호출합니다. 이후 원격 결과가 로컬 runs 폴더로 수집됩니다.",
    F_SMALL,
    COL["muted"],
    1380,
    31,
)

card_y, card_w, card_h = 1270, 455, 460
xs = [70, 572, 1074]
for x in xs:
    rounded_box(x, card_y, card_w, card_h, r=24, fill=COL["bg2"])

d.text((110, card_y + 48), "핵심 기능", fill=COL["ink"], font=F_H2)
bullet_list(
    110,
    card_y + 108,
    [
        "GUI 기반 모델 후보 선택",
        "Run Selected / Run All 자동 실행",
        "Vast AI SSH 연결과 공개키 복사",
        "Remote pre-staged dataset 사용",
        "Best / Last weight 저장 옵션",
        "학습 로그와 진행 상태 표시",
    ],
    COL["blue"],
    card_w - 80,
    32,
)
d.text((612, card_y + 48), "평가 지표와 산출물", fill=COL["ink"], font=F_H2)
bullet_list(
    612,
    card_y + 108,
    [
        "Classification: Accuracy, Macro Precision, Macro Recall, Macro F1",
        "Detection: Precision, Recall, mAP50, mAP50-95",
        "Confusion Matrix로 클래스별 오분류 확인",
        "BBox preview로 detection 결과 시각 확인",
        "timings.csv로 실행 시간 분석",
    ],
    COL["green"],
    card_w - 80,
    32,
)
d.text((1114, card_y + 48), "팀 실험에서의 장점", fill=COL["ink"], font=F_H2)
bullet_list(
    1114,
    card_y + 108,
    [
        "모델별 실행 방식 표준화",
        "실험 조건과 결과 재현 가능",
        "팀원별 담당 모델 결과 취합 용이",
        "GPU 서버 변경 시에도 같은 UI 흐름 유지",
        "최종 모델 선정 근거를 객관적으로 기록",
    ],
    COL["purple"],
    card_w - 80,
    32,
)

card2_y = 1775
rounded_box(70, card2_y, 700, 430, r=24, fill=COL["bg2"])
rounded_box(830, card2_y, 700, 430, r=24, fill=COL["bg2"])
d.text((110, card2_y + 48), "실험 운영 방식", fill=COL["ink"], font=F_H2)
bullet_list(
    110,
    card2_y + 110,
    [
        "단일 모델 실행: Run Selected로 현재 선택된 후보만 실행",
        "예약 실행: 최대 5개 후보를 Queue에 넣고 순차 실행",
        "원격 데이터셋: 서버에 미리 받은 dataset path를 지정해 업로드 시간을 줄임",
        "수집 모드: Quick / Standard 등으로 결과와 weight 다운로드 범위를 조절",
    ],
    COL["teal"],
    620,
    34,
)
d.text((870, card2_y + 48), "프로그램이 남기는 기록", fill=COL["ink"], font=F_H2)
bullet_list(
    870,
    card2_y + 110,
    [
        "summary.md: 최신 성능 지표와 실험 요약",
        "metrics/*.csv: 상세 지표, confusion matrix, 클래스별 결과",
        "previews/: 예측 이미지와 bounding box 샘플",
        "checkpoints/: best.pt / last.pt 등 학습 가중치",
        "timings.csv: upload, train, collect 단계별 시간",
    ],
    COL["orange"],
    620,
    34,
)

rounded_box(70, 2220, 1460, 115, r=24, fill=COL["dark"], outline=COL["dark"], shadow=False)
d.text((110, 2253), "핵심 가치", fill="white", font=F_H2)
draw_wrapped(
    300,
    2250,
    "모델 비교를 수동 실행이 아니라 동일 조건의 자동화된 실험 기록으로 바꿔, 최종 모델 선택의 근거를 명확하게 만든다.",
    F_BODY,
    (238, 244, 255),
    1130,
    38,
)
d.text((70, 2360), "포스터 요약 그림: 상세 구현 내역보다 모델링/평가 자동화 흐름 중심", fill=COL["muted"], font=F_TINY)

OUT.parent.mkdir(parents=True, exist_ok=True)
img.save(OUT, quality=95)
print(OUT)
