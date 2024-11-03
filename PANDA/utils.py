import pandas as pd
import os
from IPython.display import Image, display

def print_slide_details(slide, show_thumbnail=True, max_size=(600,400)):
    """Print some basic information about a slide"""
    # Generate a small image thumbnail
    if show_thumbnail:
        display(slide.get_thumbnail(size=max_size))

    # Here we compute the "pixel spacing": the physical size of a pixel in the image.
    # OpenSlide gives the resolution in centimeters so we convert this to microns.
    spacing = 1 / (float(slide.properties['tiff.XResolution']) / 10000)
    
    print(f"File id: {slide}")
    print(f"Dimensions: {slide.dimensions}")
    print(f"Microns per pixel / pixel spacing: {spacing:.3f}")
    print(f"Number of levels in the image: {slide.level_count}")
    print(f"Downsample factor per level: {slide.level_downsamples}")
    print(f"Dimensions of levels: {slide.level_dimensions}")
    
    
def load_and_filter_train_data(base_path, mask_dir):
    """
    Carrega o arquivo CSV de treinamento e as máscaras, une os dados e remove as imagens sem máscaras.

    Parâmetros:
    - base_path (str): Caminho base onde o arquivo train.csv está localizado.
    - mask_dir (str): Caminho do diretório onde as máscaras estão armazenadas.

    Retorna:
    - pd.DataFrame: DataFrame combinado e filtrado, contendo apenas imagens que têm máscaras.
    """
    # Carrega o arquivo CSV de treinamento
    train_df = pd.read_csv(f'{base_path}/train.csv')
    
    # Carrega os nomes dos arquivos de máscaras
    masks = os.listdir(mask_dir)
    masks_df = pd.Series(masks).to_frame()
    masks_df.columns = ['mask_file_name']
    
    # Extrai o ID da imagem a partir do nome do arquivo de máscara
    masks_df['image_id'] = masks_df.mask_file_name.apply(lambda x: x.split('_')[0])
    
    # Mescla o DataFrame de treinamento com o DataFrame de máscaras
    train_df = pd.merge(train_df, masks_df, on='image_id', how='outer')
    del masks_df
    
    # Imprime a quantidade de imagens sem máscara
    num_images_without_mask = len(train_df[train_df.mask_file_name.isna()])
    print(f"There are {num_images_without_mask} images without a mask.")
    
    # Remove as imagens sem máscara
    train_df = train_df[~train_df.mask_file_name.isna()]
    
    return train_df
    
