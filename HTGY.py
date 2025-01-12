import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Phần 1: Gợi ý dựa trên mô tả sản phẩm (Content-based)
def content_based_recommendation(product_description, product_descriptions, keyword=None):
    product_descriptions = product_descriptions.dropna()
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(product_descriptions["product_description"])

    # Nếu có từ khóa được cung cấp, lọc các sản phẩm có liên quan
    if keyword:
        related_to_keyword = product_descriptions[
            product_descriptions["product_description"].str.contains(keyword, case=False, na=False)
        ]
        return related_to_keyword["product_description"].tolist()  # Trả về tất cả sản phẩm liên quan

    # Nếu không có từ khóa cụ thể, thực hiện tìm kiếm dựa trên mô tả chung
    true_k = 10
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1, random_state=42)
    model.fit(X)
    Y = vectorizer.transform([product_description])
    cluster_label = model.predict(Y)[0]
    cluster_products = product_descriptions[model.labels_ == cluster_label]["product_description"].tolist()
    return cluster_products  # Trả về tất cả các sản phẩm cùng nhóm

# Phần 2: Giao diện người dùng với tkinter
def display_recommendations():
    try:
        product_descriptions = pd.read_csv('product_description.csv')  # Đọc file mô tả sản phẩm
    except FileNotFoundError:
        messagebox.showerror("Lỗi", "Không tìm thấy file dữ liệu. Vui lòng kiểm tra lại đường dẫn.")
        return

    product_description = entry_product_description.get().strip()
    keyword = entry_keyword.get().strip()  # Thêm trường nhập từ khóa

    if not product_description and not keyword:
        messagebox.showwarning("Cảnh báo", "Vui lòng nhập mô tả sản phẩm hoặc từ khóa để tìm kiếm.")
        return

    result_text.delete(1.0, tk.END)  # Xóa nội dung cũ

    recommendations_content = []

    # Gợi ý dựa trên mô tả sản phẩm (content)
    if product_description:
        recommendations_content = content_based_recommendation(product_description, product_descriptions, keyword)
        result_text.insert(tk.END, "Sản phẩm được gợi ý (Dựa trên nội dung):\n", "header")
        if recommendations_content:
            for product in recommendations_content:
                result_text.insert(tk.END, f"- {product}\n", "content")
        else:
            result_text.insert(tk.END, "Không có sản phẩm nào được gợi ý.\n", "content")

    # Hiển thị kết quả sản phẩm liên quan (từ cùng nhóm)
    if product_description or keyword:
        result_text.insert(tk.END, "\nCác sản phẩm liên quan khác:\n", "header")
        related_products = set(recommendations_content)  # Gộp cả 2 danh sách
        if related_products:
            for product in related_products:
                result_text.insert(tk.END, f"- {product}\n", "related")
        else:
            result_text.insert(tk.END, "Không có sản phẩm liên quan.\n", "related")

# Tạo giao diện
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
result_text.tag_configure("related", foreground="green")

# Khởi động giao diện
root.mainloop()
