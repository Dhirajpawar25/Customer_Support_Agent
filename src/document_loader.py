"""Document loading and parsing for knowledge base."""
import re
from pathlib import Path
from typing import List, Optional
import frontmatter
from src.models import DocumentMetadata, DocumentChunk, DocumentAuthority, DocumentStatus
from src.config import config


def parse_front_matter(content: str, source_file: str) -> DocumentMetadata:
    """Parse YAML front matter from markdown file."""
    post = frontmatter.loads(content)
    
    status_str = post.get("status", "active").lower()
    if status_str == "superseded":
        status = DocumentStatus.SUPERSEDED
    elif status_str == "draft":
        status = DocumentStatus.DRAFT
    else:
        status = DocumentStatus.ACTIVE
    
    authority_str = post.get("policy_authority", "official").lower()
    if authority_str == "internal":
        authority = DocumentAuthority.INTERNAL
    elif authority_str == "legacy":
        authority = DocumentAuthority.LEGACY
    else:
        authority = DocumentAuthority.OFFICIAL
    
    return DocumentMetadata(
        document_id=post.get("document_id", ""),
        title=post.get("title", ""),
        status=status,
        effective_date=post.get("effective_date"),
        last_reviewed=post.get("last_reviewed"),
        audience=post.get("audience", "customer"),
        policy_authority=authority,
        supersedes=post.get("supersedes"),
        source_file=source_file
    )
def split_markdown_by_headings(content: str, metadata: DocumentMetadata, chunk_size: int = 500, chunk_overlap: int = 50) -> List[DocumentChunk]:
    """Split markdown content by headings, preserving heading context."""
    chunks = []
    
    lines = content.split('\n')
    current_heading = metadata.title
    current_level = 1
    current_chunk_lines = []
    chunk_index = 0
    
    for line in lines:
        heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if heading_match:
            if current_chunk_lines:
                chunk_content = '\n'.join(current_chunk_lines).strip()
                if chunk_content:
                    chunks.append(DocumentChunk(
                        content=chunk_content,
                        metadata=metadata,
                        chunk_index=chunk_index,
                        heading=current_heading,
                        heading_level=current_level
                    ))
                    chunk_index += 1
                current_chunk_lines = []
            
            current_heading = heading_match.group(2).strip()
            current_level = len(heading_match.group(1))
        else:
            current_chunk_lines.append(line)
    
    if current_chunk_lines:
        chunk_content = '\n'.join(current_chunk_lines).strip()
        if chunk_content:
            chunks.append(DocumentChunk(
                content=chunk_content,
                metadata=metadata,
                chunk_index=chunk_index,
                heading=current_heading,
                heading_level=current_level
            ))
    
    final_chunks = []
    for chunk in chunks:
        if len(chunk.content) <= chunk_size:
            final_chunks.append(chunk)
        else:
            paragraphs = chunk.content.split('\n\n')
            current_para = []
            current_len = 0
            
            for para in paragraphs:
                para_len = len(para)
                if current_len + para_len > chunk_size and current_para:
                    final_chunks.append(DocumentChunk(
                        content='\n\n'.join(current_para).strip(),
                        metadata=chunk.metadata,
                        chunk_index=len(final_chunks),
                        heading=chunk.heading,
                        heading_level=chunk.heading_level
                    ))
                    current_para = [para]
                    current_len = para_len
                else:
                    current_para.append(para)
                    current_len += para_len
            
            if current_para:
                final_chunks.append(DocumentChunk(
                    content='\n\n'.join(current_para).strip(),
                    metadata=chunk.metadata,
                    chunk_index=len(final_chunks),
                    heading=chunk.heading,
                    heading_level=chunk.heading_level
                ))
    
    for idx, chunk in enumerate(final_chunks):
        chunk.chunk_index = idx
    
    return final_chunks


def load_knowledge_base() -> List[DocumentChunk]:
    """Load and parse all knowledge base documents."""
    all_chunks = []
    
    kb_path = config.knowledge_base_path
    for md_file in sorted(kb_path.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        metadata = parse_front_matter(content, md_file.name)
        
        post = frontmatter.loads(content)
        body_content = post.content
        
        chunks = split_markdown_by_headings(
            body_content, 
            metadata, 
            config.chunk_size, 
            config.chunk_overlap
        )
        all_chunks.extend(chunks)
    
    return all_chunks


def filter_authoritative_chunks(chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    """Filter to only authoritative, active, customer-facing chunks."""
    return [
        c for c in chunks 
        if c.metadata.is_authoritative and not c.metadata.is_superseded
    ]


def get_chunks_by_source_file(chunks: List[DocumentChunk], source_file: str) -> List[DocumentChunk]:
    """Get all chunks from a specific source file."""
    return [c for c in chunks if c.metadata.source_file == source_file]