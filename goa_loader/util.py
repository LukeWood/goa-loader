def thumbnail_to_local(base_path, thumb):
    image_path=f"{base_path}/images"
    ending = "_".join(thumb.split("/")[-5:])
    return f"{image_path}/{ending}"
