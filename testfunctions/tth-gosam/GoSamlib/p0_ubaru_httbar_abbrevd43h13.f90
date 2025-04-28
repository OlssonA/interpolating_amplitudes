module     p0_ubaru_httbar_abbrevd43h13
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_kinematics, only: epstensor
   use p0_ubaru_httbar_globalsh13
   implicit none
   private
   complex(ki), dimension(32), public :: abb43
   complex(ki), public :: R2d43
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p0_ubaru_httbar_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_color, only: TR
      use p0_ubaru_httbar_globalsl1, only: epspow
      implicit none
      abb43(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb43(2)=NC**(-1)
      abb43(3)=es12**(-1)
      abb43(4)=spak2l3**(-1)
      abb43(5)=spbl3k2**(-1)
      abb43(6)=spak2l4**(-1)
      abb43(7)=spak2l5**(-1)
      abb43(8)=sqrt(mT**2)
      abb43(9)=1.0_ki/(-mT**2+es34)
      abb43(10)=abb43(2)*c1
      abb43(10)=abb43(10)-c2
      abb43(11)=abb43(10)*abb43(1)
      abb43(12)=abb43(9)*abb43(10)
      abb43(13)=abb43(11)+abb43(12)
      abb43(14)=abb43(7)*spbl4k2
      abb43(15)=abb43(14)*abb43(8)
      abb43(16)=abb43(6)*abb43(8)
      abb43(17)=abb43(16)*spbl5k2
      abb43(15)=abb43(15)+abb43(17)
      abb43(15)=-spak1k2*abb43(15)*abb43(13)
      abb43(17)=abb43(7)*spak2l3
      abb43(18)=abb43(17)*spbl3k2
      abb43(18)=abb43(18)+spbl5k2
      abb43(18)=abb43(18)*abb43(6)
      abb43(18)=abb43(18)+abb43(14)
      abb43(19)=mT*spak1k2
      abb43(20)=-abb43(19)*abb43(18)*abb43(13)
      abb43(15)=abb43(15)+abb43(20)
      abb43(15)=abb43(15)*mT
      abb43(20)=mH**2*abb43(4)*abb43(5)
      abb43(21)=abb43(20)*spbl5k2
      abb43(22)=spbl4k2*abb43(21)
      abb43(23)=-abb43(1)*abb43(22)*abb43(10)
      abb43(22)=-abb43(22)*abb43(12)
      abb43(22)=abb43(23)+abb43(22)
      abb43(23)=abb43(22)*spak1k2
      abb43(15)=abb43(15)+abb43(23)
      abb43(23)=spbl4k2*spbl5l3
      abb43(10)=-abb43(1)*spak1l3*abb43(10)*abb43(23)
      abb43(24)=abb43(12)*spbl5k2*spbl4l3
      abb43(25)=-spak1l3*abb43(24)
      abb43(10)=abb43(10)+abb43(25)
      abb43(25)=abb43(10)+abb43(15)
      abb43(26)=2.0_ki*abb43(3)
      abb43(27)=i_*gs**4*gHT*e
      abb43(28)=abb43(27)*TR**2
      abb43(29)=abb43(26)*abb43(28)
      abb43(30)=mT**2
      abb43(26)=abb43(30)*abb43(26)
      abb43(31)=1.0_ki/3.0_ki-abb43(26)
      abb43(31)=abb43(25)*abb43(31)*abb43(29)
      abb43(32)=abb43(8)**2
      abb43(18)=abb43(19)*abb43(13)*abb43(32)*abb43(18)
      abb43(19)=spbl5k2*abb43(6)
      abb43(14)=abb43(19)+abb43(14)
      abb43(14)=spak1k2*abb43(13)*abb43(14)*abb43(8)**3
      abb43(14)=abb43(14)+abb43(18)
      abb43(14)=mT*abb43(14)
      abb43(18)=-spak1k2*abb43(32)*abb43(22)
      abb43(10)=-abb43(32)*abb43(10)
      abb43(10)=abb43(14)+abb43(18)+abb43(10)
      abb43(14)=abb43(3)*TR
      abb43(14)=abb43(14)**2
      abb43(18)=4.0_ki*abb43(14)
      abb43(18)=abb43(18)*abb43(27)
      abb43(10)=abb43(10)*abb43(18)
      abb43(15)=-abb43(15)*abb43(18)
      abb43(19)=abb43(25)*abb43(18)
      abb43(22)=abb43(23)*abb43(11)
      abb43(22)=abb43(22)+abb43(24)
      abb43(14)=2.0_ki*spak1k2*abb43(22)*abb43(14)*abb43(27)
      abb43(16)=abb43(16)*abb43(13)
      abb43(22)=mT*abb43(6)*abb43(13)
      abb43(16)=abb43(22)+abb43(16)
      abb43(16)=abb43(16)*mT
      abb43(20)=-abb43(12)*abb43(20)*spbl4k2
      abb43(16)=abb43(16)-abb43(20)
      abb43(20)=abb43(3)*spak1l3*spbk2k1
      abb43(12)=abb43(12)*spbl4l3
      abb43(22)=-abb43(12)*abb43(20)
      abb43(22)=-abb43(16)+abb43(22)
      abb43(22)=abb43(22)*abb43(29)
      abb43(23)=abb43(13)*abb43(7)*abb43(8)
      abb43(24)=mT*abb43(7)*abb43(13)
      abb43(23)=abb43(24)+abb43(23)
      abb43(23)=abb43(23)*mT
      abb43(21)=abb43(11)*abb43(21)
      abb43(21)=abb43(23)+abb43(21)
      abb43(11)=abb43(11)*spbl5l3
      abb43(20)=-abb43(20)*abb43(11)
      abb43(20)=-abb43(21)+abb43(20)
      abb43(20)=abb43(20)*abb43(29)
      abb43(13)=abb43(13)*abb43(17)*abb43(6)
      abb43(17)=-abb43(13)*abb43(28)*abb43(26)
      abb43(12)=abb43(18)*abb43(12)
      abb43(11)=abb43(18)*abb43(11)
      abb43(16)=abb43(16)*abb43(18)
      abb43(21)=abb43(21)*abb43(18)
      abb43(13)=abb43(13)*abb43(30)*abb43(18)
      R2d43=abb43(31)
      rat2 = rat2 + R2d43
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='43' value='", &
          & R2d43, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd43h13
