import requests

def get_reviews(product_name, rapidapi_key):
    url = "https://g2-data-api.p.rapidapi.com/g2-products/"
    querystring = {"product": product_name}
    headers = {
        "X-RapidAPI-Key": rapidapi_key,
        "X-RapidAPI-Host": "g2-data-api.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    reviews = response.json().get('initial_reviews', [])

    processed_reviews = []
    for review in reviews:
        reviewer_name = review['reviewer']['reviewer_name']
        review_title = review['review_title']
        positive, negative, overall = split_review_content(review['review_content'])
        processed_reviews.append({
            'title': review_title,
            'reviewer': reviewer_name,
            'positive': positive.strip(),
            'negative': negative.strip(),
            'overall': overall.strip()
        })

    return processed_reviews

def split_review_content(review_content):
    parts = review_content.split('\n\n\n')
    if len(parts) == 3:
        positive, negative, overall = parts
    else:
        positive = parts[0] if len(parts) > 0 else ""
        negative = parts[1] if len(parts) > 1 else ""
        overall = parts[2] if len(parts) > 2 else ""
    return positive, negative, overall

