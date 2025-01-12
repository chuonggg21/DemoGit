import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ========================= Phần 1: Gợi ý dựa trên mô tả sản phẩm =========================
def content_based_recommendation(product_description, product_descriptions):
    """
    Gợi ý sản phẩm dựa trên nội dung (mô tả sản phẩm).
    """
    product_descriptions = product_descriptions.dropna()
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(product_descriptions["product_description"])

    true_k = 10
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1, random_state=42)
    model.fit(X)

    # Dự đoán cụm sản phẩm dựa trên mô tả đầu vào
    Y = vectorizer.transform([product_description])
    cluster_label = model.predict(Y)[0]

    # Lọc sản phẩm thuộc cùng cụm
    cluster_products = product_descriptions[model.labels_ == cluster_label]["product_description"].tolist()
    return cluster_products

# ========================= Phần 2: Gợi ý dựa trên từ khóa =========================
def keyword_based_recommendation(keyword, product_descriptions):
    """
    Gợi ý sản phẩm dựa trên từ khóa tìm kiếm.
    """
    product_descriptions = product_descriptions.dropna()
    related_to_keyword = product_descriptions[
        product_descriptions["product_description"].str.contains(keyword, case=False, na=False)
    ]
    return related_to_keyword["product_description"].tolist()

# ========================= Phần 3: Xử lý giao diện người dùng =========================
def display_recommendations():
    try:
        product_descriptions = pd.read_csv('product_description.csv')  # Đọc file mô tả sản phẩm
    except FileNotFoundError:
        messagebox.showerror("Lỗi", "Không tìm thấy file dữ liệu. Vui lòng kiểm tra lại đường dẫn.")
        return

    product_description = entry_product_description.get().strip()
    keyword = entry_keyword.get().strip()

    if not product_description and not keyword:
        messagebox.showwarning("Cảnh báo", "Vui lòng nhập mô tả sản phẩm hoặc từ khóa để tìm kiếm.")
        return

    result_text.delete(1.0, tk.END)  # Xóa nội dung cũ

    # Tìm kiếm theo từ khóa
    if keyword:
        recommendations_keyword = keyword_based_recommendation(keyword, product_descriptions)
        result_text.insert(tk.END, "Sản phẩm được gợi ý (Dựa trên từ khóa):\n", "header")
        if recommendations_keyword:
            for product in recommendations_keyword:
                result_text.insert(tk.END, f"- {product}\n", "content")
        else:
            result_text.insert(tk.END, "Không có sản phẩm nào liên quan từ khóa.\n", "content")

    # Tìm kiếm theo mô tả sản phẩm
    if product_description:
        recommendations_content = content_based_recommendation(product_description, product_descriptions)
        result_text.insert(tk.END, "\nSản phẩm được gợi ý (Dựa trên mô tả sản phẩm):\n", "header")
        if recommendations_content:
            for product in recommendations_content:
                result_text.insert(tk.END, f"- {product}\n", "content")
        else:
            result_text.insert(tk.END, "Không có sản phẩm nào được gợi ý từ mô tả.\n", "content")

    # Tìm kiếm kết hợp (sản phẩm có cả từ khóa và mô tả)
    if keyword and product_description:
        recommendations_combined = list(set(recommendations_keyword) & set(recommendations_content))
        result_text.insert(tk.END, "\nSản phẩm được gợi ý (Kết hợp từ khóa và mô tả):\n", "header")
        if recommendations_combined:
            for product in recommendations_combined:
                result_text.insert(tk.END, f"- {product}\n", "content")
        else:
            result_text.insert(tk.END, "Không có sản phẩm nào phù hợp với cả từ khóa và mô tả.\n", "content")

# ========================= Phần 4: Giao diện người dùng Tkinter =========================
root = tk.Tk()
root.title("Gợi ý sản phẩm")

# Nhập mô tả sản phẩm
label_product_description = tk.Label(root, text="Nhập mô tả sản phẩm:")
label_product_description.pack(pady=10)

entry_product_description = tk.Entry(root, width=50)
entry_product_description.pack(pady=5)

# Nhập từ khóa tìm kiếm
label_keyword = tk.Label(root, text="Nhập từ khóa tìm kiếm (ví dụ: dưỡng da, tẩy tế bào chết, ...):")
label_keyword.pack(pady=10)

entry_keyword = tk.Entry(root, width=50)
entry_keyword.pack(pady=5)

# Nút tìm kiếm
button_search = tk.Button(root, text="Tìm kiếm", command=display_recommendations)
button_search.pack(pady=20)

# Kết quả tìm kiếm
result_text = tk.Text(root, width=70, height=15)
result_text.pack(pady=10)

# Tạo các tag cho phần hiển thị
result_text.tag_configure("header", foreground="blue", font=("Arial", 12, "bold"))
result_text.tag_configure("content", foreground="green")

# Khởi động giao diện
root.mainloop()
