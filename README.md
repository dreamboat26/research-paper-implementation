# Lightning Attention Papers

## Key Contributions
- First linear attention implementation maintaining constant training speed across sequence lengths
- Solves cumulative summation (cumsum) challenges in linear attention
- Introduces novel attention calculation strategies

## Technical Approach
- Splits attention calculation into intra-blocks and inter-blocks
- Uses conventional attention computation for intra-blocks
- Applies linear attention kernel tricks for inter-blocks
- Eliminates cumsum limitations in causal settings

## Lightning Attention-2 Paper
- Enables linear attention to realize theoretical computational benefits
- Processes tokens with linear computational complexity
- Maintains constant training speed across various sequence lengths

## Methodology
- Leverages "divide and conquer" approach
- Implements tiling technique in forward and backward procedures
- Separately handles intra-block and inter-block components
- Utilizes conventional attention for intra-blocks
- Applies linear attention kernel tricks for inter-blocks
  
## Performance Highlights
- Consistent training speed irrespective of sequence length
- Handles unlimited-length sequences
- IO-aware implementation using Triton
- Surpasses existing attention mechanisms in speed and accuracy

