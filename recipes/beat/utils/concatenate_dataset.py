import braceexpand


def concatenate_dataset(urls):
    cat_urls = []
    for url in urls:
        cat_urls = cat_urls + list(braceexpand.braceexpand(url))

    return cat_urls
