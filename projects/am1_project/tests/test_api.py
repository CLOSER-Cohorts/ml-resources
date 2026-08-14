def test_root_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "API running"
    }


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_categorise_items_success(client):

    payload = {
        "items": [
            {
                "TextLabel": "How often do you play sports?",
                "ItemCategories": "every day every week",
                "ItemType": "Question",
                #"AgencyId": "uk.alspac",
                #"HasCategories": "yes"
            },
            {
                "TextLabel": "How often do you exercise?",
                "ItemCategories": "daily weekly",
                "ItemType": "Variable",
                #"AgencyId": "uk.wchads",
                #"HasCategories": "yes"
            }
        ]
    }

    response = client.post(
        "/categorise_items/",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert "predictions" in data
    assert len(data["predictions"]) == 2

    assert data["predictions"] == [
        "topic_1",
        "topic_2"
    ]


def test_validation_error_invalid_item_type(client):

    payload = {
        "items": [
            {
                "TextLabel": "Test",
                "ItemCategories": "A B C",
                "ItemType": "invalid_type",
                "AgencyId": "uk.alspac",
                "HasCategories": "yes"
            }
        ]
    }

    response = client.post(
        "/categorise_items/",
        json=payload
    )

    assert response.status_code == 422


def test_validation_error_invalid_agency(client):

    payload = {
        "items": [
            {
                "TextLabel": "Test",
                "ItemCategories": "A B C",
                "ItemType": "question",
                "AgencyId": "bad_agency",
                "HasCategories": "yes"
            }
        ]
    }

    response = client.post(
        "/categorise_items/",
        json=payload
    )

    assert response.status_code == 422


def test_normalization_of_input_values(client):

    payload = {
        "items": [
            {
                "TextLabel": "Test",
                "ItemCategories": "A B",
                "ItemType": " Question ",
                "AgencyId": "uk.alspac",
                "HasCategories": " YES "
            }
        ]
    }

    response = client.post(
        "/categorise_items/",
        json=payload
    )

    assert response.status_code == 200