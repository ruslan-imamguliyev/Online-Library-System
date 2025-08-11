package com.project.e_library.entity;

import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

@Getter
@Setter(AccessLevel.NONE)
@Entity
@Table(name = "library")
public class Book {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "id")
    private int id;

    @Column(name = "title")
    private String title;

    @Column(name = "author")
    private String author;

    @Column(name = "genres")
    private String genre;

    @Column(name = "description")
    private String description;

    @Column(name = "coverImg")
    private String imgUrl;

    @Column(name = "likedPercent")
    private String ratingPercent;

    private float rating;

    @Column(name = "numRatings")
    private int ratingNumber;


    @Transient
    public List<String> getGenres() {
        if (genre == null) return List.of();
        String genresStr = genre.replace("[", "")
                .replace("]", "")
                .replace("'", "")
                .replace("\"", "");

        String[] genreArray = genresStr.split(",");
        return new ArrayList<>(Arrays.asList(genreArray));
    }

    @Override
    public boolean equals(Object o) {
        if (o == null || getClass() != o.getClass()) return false;
        Book book = (Book) o;
        return id == book.id && Objects.equals(title, book.title) && Objects.equals(author, book.author);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id, title, author);
    }

    @Override
    public String toString() {
        return "Book{" +
                "title='" + title + '\'' +
                ", author='" + author + '\'' +
                ", rating=" + rating +
                '}';
    }
}
