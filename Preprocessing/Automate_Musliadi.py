import pandas as pd

def preprocess(df):
    df = df.copy()
    
    # Hapus kolom yang tidak relevan
    df.drop(columns=["Name", "Cabin", "PassengerId"], inplace=True)
    
    # Isi nilai kosong pada kolom 'Age' dengan median
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # Ekstraksi bagian nomor dari tiket
    def extract_ticket_number(ticket):
        return ticket.split(" ")[-1]
    
    # Ekstraksi bagian kode/awalan dari tiket
    def extract_ticket_item(ticket):
        parts = ticket.split(" ")
        return "_".join(parts[:-1]) if len(parts) > 1 else "NONE"
    
    df["Ticket_number"] = df["Ticket"].apply(extract_ticket_number)
    df["Ticket_item"] = df["Ticket"].apply(extract_ticket_item)
    
    return df

if __name__ == "__main__":
    # Baca data mentah
    df_train = pd.read_csv("../titanic_raw/train.csv")
    df_test = pd.read_csv("../titanic_raw/test.csv")

    # Lakukan preprocessing
    preprocessed_train_df = preprocess(df_train)
    preprocessed_test_df = preprocess(df_test)

    # Simpan hasil preprocessing ke file CSV
    preprocessed_train_df.to_csv("Titanic_Preprocessed_Train.csv", index=False)
    preprocessed_test_df.to_csv("Titanic_Preprocessed_Test.csv", index=False)

    print("Preprocessing selesai. Data disimpan sebagai 'Titanic_Preprocessed_Train.csv' dan 'Titanic_Preprocessed_Test.csv'.")
