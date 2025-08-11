package com.project.e_library.repository;

import com.project.e_library.entity.Book;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.awt.print.Pageable;
import java.util.List;

@Repository
public interface BookRepo extends JpaRepository<Book,Long> {

    @Query(value = "SELECT * FROM library ORDER BY RAND() LIMIT :count", nativeQuery = true)
    List<Book> findRandomBooks(@Param("count") int count);

}
