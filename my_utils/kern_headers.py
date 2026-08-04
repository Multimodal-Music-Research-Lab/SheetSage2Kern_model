HEADERS = ["**kern\t**cdata", "*clefG2\t*clefG2"]
FOOTER = ["==\t==", "*-\t*-"]


def strip_kern_headers(text: str) -> str:
    lines = text.splitlines()

    start = 0
    for header in HEADERS:
        if start < len(lines) and lines[start].strip() == header:
            start += 1

    end = len(lines)
    for footer in reversed(FOOTER):
        if end > start and lines[end - 1].strip() == footer:
            end -= 1

    return "\n".join(lines[start:end])
