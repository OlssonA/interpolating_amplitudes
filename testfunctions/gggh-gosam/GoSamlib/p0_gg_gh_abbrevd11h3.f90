module     p0_gg_gh_abbrevd11h3
   use p0_gg_gh_config, only: ki
   use p0_gg_gh_kinematics, only: epstensor
   use p0_gg_gh_globalsh3
   implicit none
   private
   complex(ki), dimension(30), public :: abb11
   complex(ki), public :: R2d11
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p0_gg_gh_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p0_gg_gh_kinematics
      use p0_gg_gh_model
      use p0_gg_gh_color, only: TR
      use p0_gg_gh_globalsl1, only: epspow
      implicit none
      abb11(1)=sqrt(mT**2)
      abb11(2)=sqrt2**(-1)
      abb11(3)=spak2k3**(-1)
      abb11(4)=spak1k2**(-1)
      abb11(5)=spak2l4**(-1)
      abb11(6)=spbl4k2**(-1)
      abb11(7)=c1-c2
      abb11(8)=abb11(7)*abb11(4)*abb11(1)**3
      abb11(9)=i_*e*gHT*abb11(2)
      abb11(10)=abb11(9)*abb11(3)*abb11(8)
      abb11(11)=abb11(3)**2
      abb11(12)=abb11(11)*abb11(9)
      abb11(13)=-abb11(1)*abb11(7)*abb11(12)
      abb11(14)=spbl4k1*spak3l4
      abb11(15)=abb11(13)*abb11(14)
      abb11(16)=abb11(15)-abb11(10)
      abb11(17)=spbk3k1*es23
      abb11(16)=abb11(17)*abb11(16)
      abb11(10)=abb11(10)*spbk3k1
      abb11(7)=abb11(7)*abb11(9)*abb11(4)*abb11(1)
      abb11(9)=-abb11(3)*abb11(7)
      abb11(18)=mH**2
      abb11(19)=abb11(5)*abb11(18)*abb11(6)
      abb11(20)=abb11(19)*abb11(9)
      abb11(21)=-abb11(17)*abb11(20)
      abb11(21)=-2.0_ki*abb11(10)+abb11(21)
      abb11(21)=es12*abb11(21)
      abb11(22)=spak2l4*spbl4k2
      abb11(10)=abb11(22)*abb11(10)
      abb11(10)=abb11(10)+abb11(21)+abb11(16)
      abb11(10)=2.0_ki*abb11(10)
      abb11(16)=-es12*spbk3k1
      abb11(16)=abb11(17)+abb11(16)
      abb11(16)=abb11(20)*abb11(16)
      abb11(21)=spbk3k1*abb11(15)
      abb11(16)=abb11(21)+abb11(16)
      abb11(16)=4.0_ki*abb11(16)
      abb11(21)=8.0_ki*spbk3k1*abb11(20)
      abb11(23)=2.0_ki*abb11(9)
      abb11(24)=abb11(17)*abb11(23)
      abb11(17)=2.0_ki*abb11(17)
      abb11(25)=abb11(13)*abb11(17)
      abb11(26)=4.0_ki*spbk3k1
      abb11(27)=abb11(13)*abb11(26)
      abb11(7)=abb11(11)*abb11(7)
      abb11(11)=abb11(7)*spak3l4
      abb11(17)=abb11(11)*abb11(17)
      abb11(26)=abb11(11)*abb11(26)
      abb11(28)=abb11(7)*es23
      abb11(14)=-abb11(28)*abb11(14)
      abb11(20)=abb11(20)*spbk2k1
      abb11(29)=-es23*abb11(20)
      abb11(14)=abb11(14)+abb11(29)
      abb11(14)=2.0_ki*abb11(14)
      abb11(29)=-spbl4k1*abb11(11)
      abb11(20)=abb11(29)-abb11(20)
      abb11(20)=4.0_ki*abb11(20)
      abb11(9)=-4.0_ki*abb11(9)*spbk2k1
      abb11(23)=-spbl4k2*abb11(23)
      abb11(29)=4.0_ki*abb11(28)
      abb11(8)=-abb11(12)*abb11(8)
      abb11(8)=2.0_ki*abb11(8)+abb11(28)
      abb11(8)=es23*abb11(8)
      abb11(12)=abb11(28)*abb11(19)
      abb11(19)=abb11(28)+abb11(12)
      abb11(19)=es12*abb11(19)
      abb11(30)=-abb11(18)*abb11(28)
      abb11(15)=-spbk3k2*abb11(15)
      abb11(8)=abb11(15)+abb11(19)+abb11(8)+abb11(30)
      abb11(8)=2.0_ki*abb11(8)
      abb11(15)=2.0_ki*abb11(7)
      abb11(19)=es12*abb11(15)
      abb11(22)=-abb11(7)*abb11(22)
      abb11(19)=abb11(19)+abb11(22)
      abb11(19)=4.0_ki*abb11(19)
      abb11(18)=es12-abb11(18)
      abb11(18)=abb11(7)*abb11(18)
      abb11(12)=-abb11(12)+abb11(28)+abb11(18)
      abb11(12)=4.0_ki*abb11(12)
      abb11(18)=2.0_ki*spbk3k2
      abb11(13)=-abb11(13)*abb11(18)
      abb11(11)=-abb11(11)*abb11(18)
      abb11(7)=-8.0_ki*abb11(7)
      abb11(15)=-spak1k3*spbk3k2*abb11(15)
      R2d11=0.0_ki
      rat2 = rat2 + R2d11
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='11' value='", &
          & R2d11, "'/>"
      end if
   end subroutine
end module p0_gg_gh_abbrevd11h3
