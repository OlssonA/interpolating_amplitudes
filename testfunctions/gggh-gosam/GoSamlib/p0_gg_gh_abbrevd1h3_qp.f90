module     p0_gg_gh_abbrevd1h3_qp
   use p0_gg_gh_config, only: ki => ki_qp
   use p0_gg_gh_kinematics_qp, only: epstensor
   use p0_gg_gh_globalsh3_qp
   implicit none
   private
   complex(ki), dimension(25), public :: abb1
   complex(ki), public :: R2d1
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p0_gg_gh_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_model_qp
      use p0_gg_gh_color_qp, only: TR
      use p0_gg_gh_globalsl1_qp, only: epspow
      implicit none
      abb1(1)=sqrt(mT**2)
      abb1(2)=sqrt2**(-1)
      abb1(3)=spak1k2**(-1)
      abb1(4)=spak2k3**(-1)
      abb1(5)=es12**(-1)
      abb1(6)=spak2l4**(-1)
      abb1(7)=spbl4k2**(-1)
      abb1(8)=c1-c2
      abb1(9)=abb1(2)*i_*e*gHT*abb1(3)
      abb1(10)=abb1(1)*abb1(9)
      abb1(11)=abb1(8)*abb1(10)*abb1(4)
      abb1(12)=-abb1(11)*spbk3k1
      abb1(13)=2.0_ki*abb1(12)
      abb1(14)=mH**2
      abb1(15)=abb1(6)*abb1(14)*abb1(7)
      abb1(16)=abb1(12)*abb1(15)
      abb1(17)=abb1(5)*spbk2k1
      abb1(18)=abb1(17)*abb1(11)
      abb1(19)=spbl4k3*spak2l4
      abb1(20)=abb1(19)*abb1(18)
      abb1(16)=abb1(16)+abb1(20)
      abb1(20)=-es23*abb1(16)
      abb1(21)=spbk2k1*abb1(11)
      abb1(22)=-spak2l4*abb1(21)
      abb1(23)=abb1(18)*spak2l4
      abb1(24)=abb1(14)*abb1(23)
      abb1(22)=abb1(22)+abb1(24)
      abb1(22)=spbl4k3*abb1(22)
      abb1(9)=-abb1(4)*abb1(1)**3*abb1(9)*abb1(8)*spbk3k1
      abb1(9)=abb1(9)+abb1(22)
      abb1(17)=abb1(8)*abb1(17)*abb1(10)
      abb1(22)=-spbk3k1*abb1(17)
      abb1(24)=-abb1(22)*spak1l4*spbl4k3
      abb1(9)=abb1(24)+2.0_ki*abb1(9)+abb1(20)
      abb1(9)=2.0_ki*abb1(9)
      abb1(16)=-4.0_ki*abb1(16)
      abb1(12)=4.0_ki*abb1(12)
      abb1(20)=2.0_ki*spbl4k1
      abb1(11)=abb1(11)*abb1(20)
      abb1(24)=abb1(4)**2
      abb1(8)=-abb1(24)*abb1(10)*abb1(8)
      abb1(10)=spbk2k1*abb1(8)
      abb1(17)=-abb1(24)*abb1(17)
      abb1(24)=-abb1(17)*abb1(14)
      abb1(10)=abb1(10)+abb1(24)
      abb1(24)=es23*abb1(17)
      abb1(10)=2.0_ki*abb1(10)+abb1(24)
      abb1(10)=spak2l4*abb1(10)
      abb1(24)=abb1(18)*spak1l4
      abb1(25)=spbk3k1*abb1(24)
      abb1(10)=abb1(25)+abb1(10)
      abb1(10)=2.0_ki*abb1(10)
      abb1(14)=-abb1(18)*abb1(14)
      abb1(14)=abb1(14)+abb1(21)
      abb1(25)=es23*abb1(18)
      abb1(14)=2.0_ki*abb1(14)+abb1(25)
      abb1(17)=abb1(17)*spak1k3
      abb1(20)=spak2l4*abb1(17)*abb1(20)
      abb1(21)=abb1(21)*abb1(15)
      abb1(25)=-spbl4k2*abb1(23)
      abb1(14)=abb1(25)+abb1(20)+2.0_ki*abb1(14)+abb1(21)
      abb1(14)=2.0_ki*abb1(14)
      abb1(20)=16.0_ki*abb1(18)
      abb1(8)=-es23*abb1(15)*abb1(8)
      abb1(15)=abb1(19)*abb1(17)
      abb1(19)=spbl4k3*abb1(24)
      abb1(8)=2.0_ki*abb1(15)+abb1(8)+abb1(19)
      abb1(8)=2.0_ki*abb1(8)
      abb1(15)=-16.0_ki*abb1(17)
      abb1(17)=spbl4k1*abb1(23)
      abb1(17)=2.0_ki*abb1(22)+abb1(17)
      abb1(17)=2.0_ki*abb1(17)
      abb1(18)=-8.0_ki*abb1(18)
      R2d1=abb1(13)
      rat2 = rat2 + R2d1
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='1' value='", &
          & R2d1, "'/>"
      end if
   end subroutine
end module p0_gg_gh_abbrevd1h3_qp
