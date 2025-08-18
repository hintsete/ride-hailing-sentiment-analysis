# import json
# import os
# from google_play_scraper import Sort, reviews

# def scrape_reviews(app_id, lang='en', country='et', count=1000):
#     """
#     Scrape reviews from Google Play for a given app ID.
#     """
#     try:
#         result, _ = reviews(
#             app_id,
#             lang=lang,
#             country=country,
#             sort=Sort.NEWEST,
#             count=count
#         )
#         return result
#     except Exception as e:
#         print(f"Error scraping {app_id}: {e}")
#         return []

# def save_reviews():
#     """
#     Scrape and save reviews for RIDE and Feres.
#     """
#     apps = {
#         'ride': 'com.multibrains.taxi.passenger.ridepassengeret',
#         'feres': 'et.com.feres.ferespassenger'  # Verify this ID
#     }
    
#     os.makedirs('data/raw', exist_ok=True)
    
#     for app_name, app_id in apps.items():
#         reviews_data = scrape_reviews(app_id)
#         if reviews_data:
#             file_path = f'data/raw/{app_name}_reviews.json'
#             with open(file_path, 'w', encoding='utf-8') as f:
#                 json.dump(reviews_data, f, ensure_ascii=False, indent=4, default=str)
#             print(f"Saved {len(reviews_data)} reviews for {app_name} to {file_path}")
#         else:
#             print(f"No reviews scraped for {app_name}")

# if __name__ == "__main__":
#     save_reviews()
import json
import os
from google_play_scraper import Sort, reviews
from time import sleep

def scrape_reviews(app_id, lang='en', countries=['et', 'us'], total_count=1000, retry=3):
    """
    Scrape reviews from Google Play for a given app ID.
    Tries multiple countries and retries on network errors.
    Returns up to total_count reviews.
    """
    all_reviews = []
    remaining = total_count

    for country in countries:
        if remaining <= 0:
            break  # Stop if we already have enough

        attempt = 0
        while attempt < retry:
            try:
                result, _ = reviews(
                    app_id,
                    lang=lang,
                    country=country,
                    sort=Sort.NEWEST,
                    count=remaining  # only fetch what’s remaining
                )
                # Convert datetime to string
                for review in result:
                    if 'at' in review and review['at'] is not None:
                        review['at'] = review['at'].isoformat()

                all_reviews.extend(result)
                remaining = total_count - len(all_reviews)
                break  # Success, move to next country
            except Exception as e:
                attempt += 1
                print(f"Error scraping {app_id} ({country}), attempt {attempt}: {e}")
                sleep(2)

    return all_reviews

def save_reviews():
    """
    Scrape and save reviews for Ride and Feres.
    """
    apps = {
        'ride': 'com.multibrains.taxi.passenger.ridepassengeret',
        'feres': 'com.feres.user'
    }
    os.makedirs('data/raw', exist_ok=True)

    for app_name, app_id in apps.items():
        reviews_data = scrape_reviews(app_id, total_count=1000)
        if reviews_data:
            file_path = f'data/raw/{app_name}_reviews.json'
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(reviews_data, f, ensure_ascii=False, indent=4,default=str)
            print(f"Saved {len(reviews_data)} reviews for {app_name} to {file_path}")
        else:
            print(f"No reviews scraped for {app_name}")

if __name__ == "__main__":
    save_reviews()
