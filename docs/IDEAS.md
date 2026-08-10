# Идеи и расширения — НЕ трогаем, пока не закрыт S4

Правило из брифа: не добавлять модели и анализы сверх плана до закрытия S4.
Всё, что приходит в голову по дороге, складываем сюда.

## Модели
- [ ] DINOv2 с registers (ViT-B/14-reg) — влияют ли register-токены на brain alignment
- [ ] MAE (masked autoencoder) — другой тип self-supervision, реконструктивный, не контрастивный
- [ ] SigLIP / EVA-CLIP — более сильные language-supervised
- [ ] Видео-модели (VideoMAE, V-JEPA) — но NSD статичен, смысл сомнительный

## Анализы
- [ ] RDM-метрика: сравнить 1−Pearson с cross-validated Mahalanobis (crossnobis) —
      crossnobis несмещённая, это стандарт в RSA-сообществе
- [ ] Weighted RSA / voxel reweighting — модели редко выигрывают «как есть»
- [ ] Searchlight RSA по всей коре вместо ROI-подхода — красивые карты, но дорого
- [ ] Разложение: сколько alignment объясняется низкоуровневой статистикой (Gabor-модель как baseline)
- [ ] Behavioral RDM из NSD (данные о памяти / реакциях) как третья точка сравнения

## Инфраструктура
- [ ] `range_read` стратегия скачивания (см. docs/NSD_ACCESS.md §4) — 30x меньше трафика
- [ ] Перевод кэша активаций в float16 если упрёмся в диск
