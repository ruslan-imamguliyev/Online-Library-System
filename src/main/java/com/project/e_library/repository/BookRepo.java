package com.project.e_library.repository;

import com.project.e_library.entity.Book;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;


import java.util.List;

@Repository
public interface BookRepo extends JpaRepository<Book,Long> {

    @Query(value = "SELECT * FROM library ORDER BY RAND() LIMIT :count", nativeQuery = true)
    List<Book> findRandomBooks(@Param("count") int count);

    @Query("SELECT b FROM Book b WHERE " +
            "LOWER(b.title) LIKE LOWER(CONCAT('%', :keyword, '%')) OR " +
            "LOWER(b.author) LIKE LOWER(CONCAT('%', :keyword, '%')) OR " +
            "LOWER(b.description) LIKE LOWER(CONCAT('%', :keyword, '%')) " +
            "ORDER BY " +
            "CASE WHEN LOWER(b.title) = LOWER(:keyword) THEN 0 " +
            "WHEN LOWER(b.title) LIKE LOWER(CONCAT('%', :keyword, '%')) THEN 1 " +
            "WHEN LOWER(b.author) = LOWER(:keyword) THEN 2 " +
            "WHEN LOWER(b.author) = LOWER(CONCAT('%', :keyword, '%')) THEN 3 "+
            "ELSE 4 END")
    List<Book> searchBook(@Param("keyword") String keyword,Pageable pageable);


    @Query("SELECT DISTINCT b FROM Book b "+
            "JOIN b.genres g "+
            "WHERE LOWER(g) IN :genres")
    List<Book> filterByGenres(@Param("genres") List<String> genres, Pageable pageable);

}
