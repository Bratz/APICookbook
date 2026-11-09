"""GET /api/endpoint/{name}: the sidebar's and recipes' endpoint card, straight from the spec."""


def test_a_swagger_operation_has_its_parameters_and_a_complete_curl(client):
    card = client.get("/api/endpoint/RSTETViewCustomerDetailsUsingGET").json()

    assert card["method"] == "GET"
    assert card["path"] == "/Core/customerManagement/customer/viewDetails"
    assert card["parameters"]["CustomerID"]["required"] is True
    # The required query parameter is in the URL, so the copied curl works once it is filled in.
    assert card["url"] == "/Core/customerManagement/customer/viewDetails?CustomerID={CustomerID}"
    assert "viewDetails?CustomerID={CustomerID}" in card["curl"]
    assert all(h in card["curl"] for h in ("entity:", "userId:", "languageCode:"))
    assert card["example_body"] is None


def test_a_write_has_its_body_schema_and_an_example_body(client):
    card = client.get("/api/endpoint/RSTBTAmendPhonenumberUsingPUT").json()

    assert card["method"] == "PUT" and card["body_schema"]
    assert "customerContactDetails" in card["example_body"]
    assert "-d '" in card["curl"] and "Content-Type: application/json" in card["curl"]


def test_a_collection_nickname_is_the_endpoint_it_names(client):
    # The search chat used to answer "GetAccountBalance" with a different endpoint.
    card = client.get("/api/endpoint/GetAccountBalance").json()

    assert card["name"] == "GetAccountBalance"
    assert card["path"].startswith("/Core/accountManagement/account/balanceDetails/")


def test_an_unknown_name_is_a_404(client):
    assert client.get("/api/endpoint/NoSuchOperation").status_code == 404
