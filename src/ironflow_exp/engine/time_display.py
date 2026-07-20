from datetime import datetime, timedelta, timezone


KST = timezone(timedelta(hours=9), name='KST')


def iso_to_kst_display(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None

    normalized = value.strip()
    if normalized.endswith('Z'):
        normalized = f'{normalized[:-1]}+00:00'

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(KST).strftime('%Y-%m-%d %H:%M:%S KST')


def add_kst_display_fields(data: dict[str, object], fields: tuple[str, ...]) -> dict[str, object]:
    updated = dict(data)
    for field in fields:
        value = updated.get(field)
        if isinstance(value, str):
            kst_value = iso_to_kst_display(value=value)
            if kst_value is not None:
                updated[f'{field}_kst'] = kst_value

    return updated
