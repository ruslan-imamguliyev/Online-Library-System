package com.project.e_library.service;

import com.project.e_library.entity.Book;
import com.project.e_library.repository.BookRepo;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class BookService {

    private final BookRepo bookRepo;

    @Autowired
    public BookService(BookRepo bookRepo) {
        this.bookRepo = bookRepo;
    }

    public List<Book> getRandBooks() {
        return bookRepo.findRandomBooks(10);
    }

    public List<Book> searchBook(String keyword, int page, int size) {
        return bookRepo.searchBook(keyword, PageRequest.of(page, size));
    }

    public List<Book> filterBookByGenre(List<String> genres, int page, int size) {
        Pageable pageable = PageRequest.of(page, size);
        return bookRepo.filterByGenres(genres,pageable);
    }
}
