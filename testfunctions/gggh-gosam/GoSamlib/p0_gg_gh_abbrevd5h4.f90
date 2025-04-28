module     p0_gg_gh_abbrevd5h4
   use p0_gg_gh_config, only: ki
   use p0_gg_gh_kinematics, only: epstensor
   use p0_gg_gh_globalsh4
   implicit none
   private
   complex(ki), dimension(20), public :: abb5
   complex(ki), public :: R2d5
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
      abb5(1)=sqrt(mT**2)
      abb5(2)=sqrt2**(-1)
      abb5(3)=spbk2k1**(-1)
      abb5(4)=spbk3k2**(-1)
      abb5(5)=spak2l4**(-1)
      abb5(6)=spbl4k2**(-1)
      abb5(7)=c1-c2
      abb5(8)=i_*e*gHT*abb5(2)
      abb5(9)=abb5(7)*abb5(1)*abb5(8)*abb5(3)
      abb5(10)=-abb5(4)*abb5(9)
      abb5(11)=2.0_ki*spak1k3
      abb5(12)=abb5(10)*abb5(11)
      abb5(13)=-abb5(11)*abb5(3)*abb5(4)*abb5(1)**3*abb5(7)*abb5(8)
      abb5(14)=abb5(4)**2
      abb5(7)=-abb5(8)*abb5(1)*abb5(7)*abb5(14)
      abb5(8)=spbl4k3*spak1k3
      abb5(15)=abb5(7)*spak1l4*abb5(8)
      abb5(9)=abb5(14)*abb5(9)
      abb5(14)=abb5(9)*spak1l4
      abb5(16)=abb5(14)*spbl4k2
      abb5(17)=mH**2*abb5(6)*abb5(5)
      abb5(18)=abb5(10)*abb5(17)
      abb5(19)=spak1k3*abb5(18)
      abb5(20)=-abb5(19)-abb5(16)
      abb5(20)=es12*abb5(20)
      abb5(13)=abb5(20)+abb5(13)+abb5(15)
      abb5(13)=2.0_ki*abb5(13)
      abb5(15)=-abb5(19)+abb5(16)
      abb5(15)=4.0_ki*abb5(15)
      abb5(16)=4.0_ki*abb5(10)*spak1k3
      abb5(19)=2.0_ki*abb5(9)
      abb5(20)=-spak1k2*spbl4k2
      abb5(8)=-abb5(8)+abb5(20)
      abb5(8)=abb5(8)*abb5(19)
      abb5(14)=spbl4k3*abb5(14)
      abb5(18)=spak1k2*abb5(18)
      abb5(14)=abb5(14)+abb5(18)
      abb5(14)=2.0_ki*abb5(14)
      abb5(10)=2.0_ki*spak3l4*abb5(10)
      abb5(7)=-abb5(7)*abb5(11)
      abb5(11)=spak3l4*abb5(9)*spbl4k2
      abb5(7)=abb5(7)+abb5(11)
      abb5(7)=2.0_ki*abb5(7)
      abb5(11)=-8.0_ki*abb5(9)
      abb5(18)=spak2l4*spbl4k2
      abb5(17)=-es23*abb5(17)
      abb5(17)=abb5(17)+2.0_ki*es12+abb5(18)
      abb5(17)=abb5(17)*abb5(19)
      abb5(9)=-16.0_ki*abb5(9)
      R2d5=abb5(12)
      rat2 = rat2 + R2d5
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='5' value='", &
          & R2d5, "'/>"
      end if
   end subroutine
end module p0_gg_gh_abbrevd5h4
