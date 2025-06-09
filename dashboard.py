import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prediction import predict_attrition

# Load data untuk dashboard
@st.cache_data
def load_data():
    return pd.read_csv('datasets/attrition_data.csv')

df = load_data()

# Sidebar menu
menu = st.sidebar.selectbox(
    'Pilih Menu',
    ['Dashboard Utama', 'Analisis Attrition', 'Prediksi Karyawan', 'Tentang Proyek']
)

st.title('HR Attrition Dashboard')

if menu == 'Dashboard Utama':
    st.header('Statistik Umum')
    st.markdown('''
    Dashboard ini menyajikan gambaran umum kondisi karyawan di perusahaan Jaya Jaya Maju, termasuk proporsi attrition (keluar), distribusi usia, pendapatan, serta faktor risiko utama. Visualisasi dan insight di bawah ini membantu HR memahami pola dan kelompok karyawan yang perlu perhatian lebih untuk mencegah attrition.
    ''')
    col1, col2, col3 = st.columns(3)
    col1.metric('👥 Jumlah Karyawan', len(df))
    col2.metric('📉 Attrition Rate (%)', f"{df['Attrition'].mean()*100:.2f}")
    col3.metric('🎂 Rata-rata Usia', f"{df['Age'].mean():.1f}")

    # Pie chart komposisi attrition
    attr_pie = df['Attrition'].value_counts(normalize=True).rename({0:'Bertahan',1:'Keluar'})
    fig_pie = px.pie(names=attr_pie.index, values=attr_pie.values, color=attr_pie.index,
                    color_discrete_map={'Bertahan':'#00cc96','Keluar':'#ef553b'},
                    title='Komposisi Karyawan Bertahan vs Keluar')
    st.plotly_chart(fig_pie, use_container_width=True)
    st.caption('''🔎 **Insight:** Mayoritas karyawan masih bertahan, namun proporsi keluar perlu diwaspadai.''')

    # Histogram usia
    st.plotly_chart(px.histogram(df, x='Age', color='Attrition', barmode='overlay',
                                 color_discrete_map={0:'#00cc96',1:'#ef553b'},
                                 title='Distribusi Usia Karyawan berdasarkan Attrition'), use_container_width=True)
    st.caption('''🧑‍💼 **Insight:** Karyawan yang keluar didominasi usia muda-pertengahan.''')

    # Boxplot income
    st.plotly_chart(px.box(df, x='Attrition', y='MonthlyIncome',
                          color='Attrition',
                          color_discrete_map={0:'#00cc96',1:'#ef553b'},
                          title='Monthly Income berdasarkan Attrition'), use_container_width=True)
    st.caption('''💸 **Insight:** Karyawan dengan pendapatan lebih rendah cenderung keluar.''')

    # Penjelasan insight umum di bawah visualisasi
    st.markdown('''
    ---
    **Kesimpulan Dashboard Utama:**
    - Mayoritas karyawan yang keluar (attrition) berada pada rentang usia muda hingga pertengahan.
    - Karyawan dengan pendapatan bulanan lebih rendah dan yang sering lembur cenderung memiliki risiko keluar yang lebih tinggi.
    - Visualisasi ini membantu HR untuk mengidentifikasi kelompok yang perlu perhatian lebih dalam upaya retensi karyawan.
    ''')

elif menu == 'Analisis Attrition':
    st.header('Analisis Faktor Attrition')
    st.markdown('''
    Pada menu ini, Anda dapat melihat visualisasi dan statistik rata-rata attrition berdasarkan beberapa faktor utama:
    - **Department**: Menampilkan perbandingan tingkat attrition di setiap departemen.
    - **Education Field**: Menunjukkan bidang pendidikan mana yang memiliki risiko attrition lebih tinggi.
    - **Job Level**: Memvisualisasikan hubungan antara level jabatan dan tingkat attrition.
    - **Marital Status**: Menampilkan perbedaan attrition berdasarkan status pernikahan karyawan.
    
    Setiap grafik dilengkapi dengan nilai persentase attrition untuk memudahkan interpretasi dan pengambilan keputusan oleh tim HR.
    ''')
    # Reverse one-hot encoding untuk visualisasi
    df_viz = df.copy()
    if 'Department_Research & Development' in df_viz.columns:
        df_viz['Department'] = np.select(
            [
                df_viz.get('Department_Research & Development', 0) == 1,
                df_viz.get('Department_Sales', 0) == 1
            ],
            ['Research & Development', 'Sales'],
            default='Human Resources'
        )
    if 'OverTime_Yes' in df_viz.columns:
        df_viz['OverTime'] = df_viz['OverTime_Yes'].map({1: 'Yes', 0: 'No'})
    if 'MaritalStatus_Single' in df_viz.columns:
        df_viz['MaritalStatus'] = np.select(
            [
                df_viz.get('MaritalStatus_Single', 0) == 1,
                df_viz.get('MaritalStatus_Married', 0) == 1
            ],
            ['Single', 'Married'],
            default='Divorced'
        )
    st.plotly_chart(px.bar(df_viz.groupby('Department')['Attrition'].mean().reset_index(),
                          x='Department', y='Attrition',
                          title='Rata-rata Attrition per Department'))
    # Tampilkan nilai rata-rata attrition per Department
    dept_stats = df_viz.groupby('Department')['Attrition'].mean().reset_index()
    for _, row in dept_stats.iterrows():
        st.write(f"Department {row['Department']} Attrition Rate = {row['Attrition']*100:.2f}%")

    if 'EducationField_Life Sciences' in df_viz.columns:
        df_viz['EducationField'] = np.select(
            [
                df_viz.get('EducationField_Life Sciences', 0) == 1,
                df_viz.get('EducationField_Medical', 0) == 1,
                df_viz.get('EducationField_Marketing', 0) == 1,
                df_viz.get('EducationField_Technical Degree', 0) == 1,
                df_viz.get('EducationField_Other', 0) == 1
            ],
            ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other'],
            default='Human Resources'
        )
        st.plotly_chart(px.bar(df_viz.groupby('EducationField')['Attrition'].mean().reset_index(),
                              x='EducationField', y='Attrition',
                              title='Attrition berdasarkan Education Field'))
        # Tampilkan nilai rata-rata attrition per EducationField
        edu_stats = df_viz.groupby('EducationField')['Attrition'].mean().reset_index()
        for _, row in edu_stats.iterrows():
            st.write(f"Education Field {row['EducationField']} Attrition Rate = {row['Attrition']*100:.2f}%")

    st.plotly_chart(px.bar(df_viz.groupby('JobLevel')['Attrition'].mean().reset_index(),
                          x='JobLevel', y='Attrition',
                          title='Attrition berdasarkan Job Level'))
    # Tampilkan nilai rata-rata attrition per JobLevel
    job_stats = df_viz.groupby('JobLevel')['Attrition'].mean().reset_index()
    for _, row in job_stats.iterrows():
        st.write(f"Job Level {row['JobLevel']} Attrition Rate = {row['Attrition']*100:.2f}%")

    st.plotly_chart(px.bar(df_viz.groupby('MaritalStatus')['Attrition'].mean().reset_index(),
                          x='MaritalStatus', y='Attrition',
                          title='Attrition berdasarkan Status Pernikahan'))
    # Tampilkan nilai rata-rata attrition per MaritalStatus
    marital_stats = df_viz.groupby('MaritalStatus')['Attrition'].mean().reset_index()
    for _, row in marital_stats.iterrows():
        st.write(f"Marital Status {row['MaritalStatus']} Attrition Rate = {row['Attrition']*100:.2f}%")

elif menu == 'Prediksi Karyawan':
    st.header('Prediksi Risiko Karyawan Keluar')
    st.markdown('''
    Pada menu ini, Anda dapat melakukan simulasi prediksi risiko attrition (keluar) untuk seorang karyawan berdasarkan data individual. Masukkan data karyawan pada form di bawah, lalu sistem akan memproses dan menampilkan prediksi status serta probabilitas risiko keluar. Fitur ini membantu HR dalam melakukan deteksi dini dan intervensi pada karyawan yang berpotensi keluar.
    ''')
    st.write('''Masukkan data karyawan di bawah ini untuk memprediksi kemungkinan keluar dari perusahaan. 
Hasil prediksi akan menampilkan status dan probabilitas risiko attrition.''')
    with st.form('form_prediksi'):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider('Usia', 18, 60, 30)
            daily_rate = st.number_input('Daily Rate', 100, 1500, 800)
            distance = st.slider('Distance From Home', 1, 30, 5)
            education = st.selectbox('Education', [1,2,3,4,5], format_func=lambda x: f'Level {x}')
            env_sat = st.selectbox('Environment Satisfaction', [1,2,3,4], format_func=lambda x: f'Level {x}')
        with col2:
            overtime = st.radio('OverTime', ['No', 'Yes'])
            marital = st.radio('Marital Status', ['Married', 'Single'])
            joblevel = st.selectbox('Job Level', [1,2,3,4,5], format_func=lambda x: f'Level {x}')
            monthly_income = st.number_input('Monthly Income', 1000, 20000, 5000)
        submitted = st.form_submit_button('Prediksi')
    # One-hot encoding manual (harus sesuai fitur model)
    input_dict = {
        'Age': age,
        'DailyRate': daily_rate,
        'DistanceFromHome': distance,
        'Education': education,
        'EnvironmentSatisfaction': env_sat,
        'JobLevel': joblevel,
        'MonthlyIncome': monthly_income,
        'OverTime_Yes': 1 if overtime=='Yes' else 0,
        'MaritalStatus_Single': 1 if marital=='Single' else 0,
        'MaritalStatus_Married': 1 if marital=='Married' else 0,
        # Tambahkan fitur lain sesuai kebutuhan dan hasil one-hot encoding
    }
    if submitted:
        pred, prob = predict_attrition(input_dict)
        st.markdown('---')
        if pred == 1:
            st.error(f'⚠️ Karyawan berisiko keluar!\nProbabilitas keluar: **{prob:.2%}**')
            st.info('Disarankan untuk melakukan monitoring dan intervensi lebih lanjut.')
        else:
            st.success(f'✅ Karyawan diprediksi bertahan.\nProbabilitas keluar: **{prob:.2%}**')
            st.info('Tetap lakukan engagement agar karyawan loyal.')

elif menu == 'Tentang Proyek':
    st.header('Tentang Proyek')
    st.markdown('''
    **Proyek Data Science: Menyelesaikan Permasalahan Human Resources**  
    Dashboard ini membantu HR dalam memahami faktor-faktor yang mempengaruhi attrition (keluarnya karyawan) dan memonitor data karyawan.  
    
    **Fitur utama dashboard:**
    - Statistik dan visualisasi attrition
    - Analisis faktor-faktor utama
    - Prediksi risiko karyawan keluar
    
    ---
    
    ### Kesimpulan
    Dashboard interaktif yang dibangun dalam proyek ini terdiri dari beberapa fitur utama:
    - **Dashboard Utama:** Menampilkan statistik dan visualisasi komprehensif mengenai kondisi karyawan, proporsi attrition, distribusi usia, pendapatan, serta insight faktor risiko utama secara ringkas.
    - **Analisis Attrition:** Menyediakan analisis mendalam dan visualisasi interaktif untuk faktor-faktor utama penyebab attrition, seperti Department, Education Field, Job Level, dan Marital Status, lengkap dengan persentase attrition tiap kategori.
    - **Prediksi Karyawan:** Fitur simulasi prediksi risiko resign untuk karyawan individual, sehingga HR dapat melakukan deteksi dini dan intervensi secara personal.
    
    Dashboard ini memudahkan tim HR untuk:
    - Memantau kondisi dan tren attrition secara real-time.
    - Mengidentifikasi kelompok karyawan yang berisiko tinggi keluar.
    - Melakukan prediksi dan intervensi berbasis data untuk meningkatkan retensi karyawan.
    
    Model prediksi attrition berbasis Logistic Regression mampu mengidentifikasi karyawan berisiko resign dengan akurasi 85%.
    Faktor utama yang memengaruhi attrition adalah Job Level, Marital Status, dan Education Field.
    HR disarankan untuk fokus pada karyawan level 1 dan status single, serta bidang pendidikan tertentu yang memiliki risiko tinggi.
    Dashboard interaktif memudahkan monitoring dan pengambilan keputusan berbasis data.
    
    ---
    
    Dibuat oleh: Yoga Samudra
    ''')
