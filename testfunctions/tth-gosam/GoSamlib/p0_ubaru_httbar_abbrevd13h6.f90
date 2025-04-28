module     p0_ubaru_httbar_abbrevd13h6
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_kinematics, only: epstensor
   use p0_ubaru_httbar_globalsh6
   implicit none
   private
   complex(ki), dimension(23), public :: abb13
   complex(ki), public :: R2d13
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
      abb13(1)=sqrt(mT**2)
      abb13(2)=NC**(-1)
      abb13(3)=es12**(-1)
      abb13(4)=es45**(-1)
      abb13(5)=spak2l3**(-1)
      abb13(6)=spbl3k2**(-1)
      abb13(7)=spak2l4**(-1)
      abb13(8)=spbl5k2**(-1)
      abb13(9)=c1*abb13(2)
      abb13(9)=abb13(9)-c2
      abb13(9)=abb13(9)*i_*e*gHT*abb13(4)*TR**2*gs**4
      abb13(10)=abb13(9)*abb13(3)
      abb13(11)=-abb13(1)*abb13(10)
      abb13(12)=spbl4k1*spak2l5
      abb13(13)=abb13(11)*abb13(12)
      abb13(14)=-4.0_ki*abb13(13)
      abb13(9)=abb13(9)*abb13(1)
      abb13(15)=mH**2*abb13(6)*abb13(5)
      abb13(16)=-abb13(9)*abb13(15)
      abb13(10)=abb13(1)**3*abb13(10)
      abb13(10)=2.0_ki*abb13(10)+abb13(16)
      abb13(10)=abb13(12)*abb13(10)
      abb13(16)=spal3l5*spbl3k1
      abb13(17)=spak1k2*spbl4k1
      abb13(18)=abb13(17)*abb13(16)
      abb13(19)=spbl4l3*spak2l3
      abb13(20)=-abb13(19)*spbk2k1*spak2l5
      abb13(18)=abb13(18)+abb13(20)
      abb13(18)=abb13(11)*abb13(18)
      abb13(20)=abb13(7)*abb13(8)*mT**2
      abb13(21)=abb13(20)*spak2l3
      abb13(22)=abb13(21)*spbl3k1
      abb13(23)=-abb13(9)*abb13(22)
      abb13(10)=abb13(23)+abb13(18)+abb13(10)
      abb13(10)=4.0_ki*abb13(10)
      abb13(12)=-abb13(12)*abb13(15)
      abb13(12)=abb13(12)+abb13(22)
      abb13(12)=8.0_ki*abb13(11)*abb13(12)
      abb13(13)=8.0_ki*abb13(13)
      abb13(17)=abb13(19)-2.0_ki*abb13(17)
      abb13(18)=4.0_ki*abb13(11)
      abb13(17)=abb13(17)*abb13(18)
      abb13(19)=abb13(11)*spak2l5
      abb13(22)=-4.0_ki*spbl4l3*abb13(19)
      abb13(23)=2.0_ki+abb13(15)
      abb13(23)=spbk2k1*abb13(23)*abb13(19)
      abb13(16)=abb13(11)*abb13(16)
      abb13(16)=abb13(23)+abb13(16)
      abb13(16)=4.0_ki*abb13(16)
      abb13(23)=spal3l5*spbl4k1
      abb13(21)=-spbk2k1*abb13(21)
      abb13(21)=abb13(21)-abb13(23)
      abb13(18)=abb13(21)*abb13(18)
      abb13(21)=spbl3k2*spak2l3*abb13(11)
      abb13(9)=abb13(21)-2.0_ki*abb13(9)
      abb13(9)=abb13(20)*abb13(9)
      abb13(15)=-spbl4k2*abb13(15)*abb13(19)
      abb13(9)=abb13(15)+abb13(9)
      abb13(9)=4.0_ki*abb13(9)
      abb13(15)=32.0_ki*abb13(11)*abb13(20)
      abb13(11)=-16.0_ki*abb13(11)
      R2d13=abb13(14)
      rat2 = rat2 + R2d13
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='13' value='", &
          & R2d13, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd13h6
