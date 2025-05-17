import pickle
import streamlit as st

model = pickle.load(open('spam.pkl', 'rb'))
cv = pickle.load(open('vectorizer.pkl', 'rb'))


def main():
    st.title('SPAM PREDICTOR')
    st.subheader('Built with Streamlit and Python')

    msg = st.text_area('ENTER TEXT')

    if st.button('PREDICT'):
        # Transform the input message using the vectorizer
        df = [msg]
        vect = cv.transform(df).toarray()

        # Predict using the loaded model
        prediction = model.predict(vect)
        result = prediction[0]

        if result == 1:
            st.error('SPAM MESSAGE')
        else:
            st.success('HAM MESSAGE')


if __name__ == '__main__':
    main()
