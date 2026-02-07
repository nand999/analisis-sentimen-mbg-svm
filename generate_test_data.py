import pandas as pd

# Test 1: Valid CSV dengan 100 baris
texts_100 = [
    "MBG program yang sangat bermanfaat",
    "Makanan bergizi gratis mantap sekali",
    "Program ini membantu anak-anak",
    "Sangat bagus untuk generasi emas",
    "Pelayanan memuaskan",
] * 10 + [
    "Program ini buruk sekali",
    "Tidak efektif sama sekali",
    "Buang-buang anggaran negara",
    "Kualitas makanan jelek",
    "Pelaksanaan tidak baik",
] * 10

pd.DataFrame({'text': texts_100}).to_csv('test_valid.csv', index=False)

# Test 2: CSV tanpa kolom text
pd.DataFrame({
    'komentar': ['Bagus', 'Buruk', 'Lumayan'],
    'rating': [5, 1, 3]
}).to_csv('test_no_text_column.csv', index=False)

# Test 3: CSV kosong
pd.DataFrame({'text': []}).to_csv('test_empty.csv', index=False)

# Test 4: CSV dengan tanggal
pd.DataFrame({
    'text': texts_100[:20],
    'created_at': pd.date_range('2024-01-01', periods=20)
}).to_csv('test_with_dates.csv', index=False)

# Test 5: CSV dengan tanggal invalid
pd.DataFrame({
    'text': texts_100[:10],
    'tanggal': ['invalid'] * 10
}).to_csv('test_invalid_dates.csv', index=False)

# Test 6: CSV 1000 baris
texts_1000 = texts_100 * 10
pd.DataFrame({'text': texts_1000}).to_csv('test_1000.csv', index=False)

# Test 7: CSV 10000 baris
texts_10000 = texts_100 * 100
pd.DataFrame({'text': texts_10000}).to_csv('test_10000.csv', index=False)

# Test 8: CSV hanya positif
texts_positive = [
    "MBG program yang sangat bermanfaat",
    "Makanan bergizi gratis mantap sekali",
    "Program ini membantu anak-anak",
    "Sangat bagus untuk generasi emas",
] * 25
pd.DataFrame({'text': texts_positive}).to_csv('test_only_positive.csv', index=False)

# Test 9: CSV hanya negatif
texts_negative = [
    "Program ini buruk sekali",
    "Tidak efektif sama sekali",
    "Buang-buang anggaran negara",
    "Kualitas makanan jelek",
] * 25
pd.DataFrame({'text': texts_negative}).to_csv('test_only_negative.csv', index=False)

print("✅ Semua file test CSV berhasil dibuat!")
print("\nFile yang dibuat:")
print("1. test_valid.csv (100 baris)")
print("2. test_no_text_column.csv")
print("3. test_empty.csv")
print("4. test_with_dates.csv")
print("5. test_invalid_dates.csv")
print("6. test_1000.csv (1000 baris)")
print("7. test_10000.csv (10000 baris)")
print("8. test_only_positive.csv")
print("9. test_only_negative.csv")
