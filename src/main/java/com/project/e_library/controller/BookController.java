package com.project.e_library.controller;

import com.project.e_library.entity.Book;
import com.project.e_library.service.BookService;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.web.csrf.CsrfToken;
import org.springframework.web.bind.annotation.*;


import java.util.List;


@RestController
@RequestMapping("/api")
public class BookController {

    private final BookService bookService;

    @Autowired
    public BookController(BookService bookService) {
        this.bookService = bookService;
    }

    @GetMapping("/books")
    public ResponseEntity<List<Book>> getRandBooks() {
        List<Book> books = bookService.getRandBooks();
        return new ResponseEntity<>(books, HttpStatus.OK);
    }

    @GetMapping("/books/search")
    public ResponseEntity<List<Book>> searchBooks(@RequestParam String keyword,
                                                  @RequestParam(defaultValue = "0")int page,
                                                  @RequestParam(defaultValue = "10")int size) {
        List<Book> books = bookService.searchBook(keyword,page,size);
        return new ResponseEntity<>(books, HttpStatus.OK);
    }

    @PostMapping("/books/filter-genre")
    public ResponseEntity<List<Book>> filterByGenres(@RequestBody List<String> genres,
                                                     @RequestParam(defaultValue = "0")int page,
                                                     @RequestParam(defaultValue = "10")int size) {
        List<Book> books = bookService.filterBookByGenre(genres,page,size);
        return new ResponseEntity<>(books, HttpStatus.OK);
    }
}
