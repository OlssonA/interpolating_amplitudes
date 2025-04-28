module     p0_ubaru_httbar_abbrevd2h13
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_kinematics, only: epstensor
   use p0_ubaru_httbar_globalsh13
   implicit none
   private
   complex(ki), dimension(51), public :: abb2
   complex(ki), public :: R2d2
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
      abb2(1)=1.0_ki/(-mT**2+es34)
      abb2(2)=es12**(-1)
      abb2(3)=spak2l3**(-1)
      abb2(4)=spbl3k2**(-1)
      abb2(5)=spak2l4**(-1)
      abb2(6)=spak2l5**(-1)
      abb2(7)=sqrt(mT**2)
      abb2(8)=spbl5k2**(-1)
      abb2(9)=NC*c2
      abb2(10)=c1-abb2(9)
      abb2(11)=i_*e*gHT*abb2(1)*TR**2*gs**4
      abb2(10)=abb2(10)*abb2(11)*spak1k2*abb2(2)
      abb2(12)=spbl4k2*mH**2*abb2(4)*abb2(3)
      abb2(13)=abb2(10)*abb2(12)
      abb2(14)=abb2(13)*spbl5k2
      abb2(15)=mT**2
      abb2(16)=-abb2(15)*abb2(10)
      abb2(17)=abb2(5)*spbl5k2
      abb2(18)=abb2(16)*abb2(17)
      abb2(19)=-mT*abb2(10)
      abb2(20)=abb2(17)*abb2(7)
      abb2(21)=abb2(20)*abb2(19)
      abb2(14)=-abb2(14)+abb2(18)+abb2(21)
      abb2(9)=abb2(11)*abb2(9)
      abb2(11)=abb2(11)*c1
      abb2(22)=abb2(9)-abb2(11)
      abb2(23)=abb2(2)*abb2(22)
      abb2(24)=abb2(23)*spbl4l3
      abb2(25)=abb2(24)*spak1l3
      abb2(26)=abb2(25)*spbl5k2
      abb2(26)=abb2(26)+abb2(14)
      abb2(27)=abb2(7)*abb2(6)
      abb2(28)=abb2(19)*abb2(27)
      abb2(29)=abb2(16)*abb2(6)
      abb2(30)=abb2(28)+abb2(29)
      abb2(31)=abb2(30)*spbl4k2
      abb2(32)=spak2l3*abb2(5)
      abb2(33)=abb2(32)*spbl3k2
      abb2(34)=abb2(33)*abb2(29)
      abb2(35)=abb2(31)+abb2(34)
      abb2(36)=abb2(26)+abb2(35)
      abb2(37)=abb2(16)*abb2(5)
      abb2(38)=abb2(7)*abb2(5)
      abb2(39)=abb2(19)*abb2(38)
      abb2(13)=abb2(25)-abb2(13)+abb2(37)+abb2(39)
      abb2(25)=spbl5k2**2
      abb2(37)=spak2l5*abb2(25)*abb2(13)
      abb2(39)=abb2(9)*spak1k2
      abb2(40)=abb2(11)*spak1k2
      abb2(39)=abb2(39)-abb2(40)
      abb2(41)=mT**4
      abb2(42)=-abb2(41)*abb2(39)
      abb2(43)=mT**3
      abb2(44)=abb2(39)*abb2(43)
      abb2(45)=abb2(7)*abb2(44)
      abb2(45)=-abb2(42)+abb2(45)
      abb2(46)=abb2(6)**2
      abb2(47)=abb2(46)*spbl4k2
      abb2(45)=abb2(45)*abb2(47)
      abb2(46)=abb2(46)*abb2(33)
      abb2(48)=-abb2(42)*abb2(46)
      abb2(45)=abb2(45)+abb2(48)
      abb2(45)=abb2(8)*abb2(45)
      abb2(37)=abb2(45)+abb2(37)
      abb2(31)=2.0_ki*abb2(31)+2.0_ki*abb2(34)
      abb2(14)=abb2(31)+abb2(14)
      abb2(14)=spbl5k1*abb2(14)
      abb2(34)=abb2(30)*spbl5k2
      abb2(45)=-spbl4k1*abb2(34)
      abb2(14)=abb2(45)+abb2(14)
      abb2(14)=spak1l5*abb2(14)
      abb2(18)=-abb2(18)+3.0_ki*abb2(21)
      abb2(18)=spbl3k2*abb2(18)
      abb2(21)=abb2(24)*spak1l5
      abb2(25)=-abb2(25)*abb2(21)
      abb2(45)=abb2(29)*abb2(17)
      abb2(48)=-spbl3k1*spak1l5*abb2(45)
      abb2(18)=abb2(48)+abb2(18)+abb2(25)
      abb2(18)=spak2l3*abb2(18)
      abb2(25)=abb2(6)*abb2(44)
      abb2(39)=abb2(39)*abb2(15)
      abb2(44)=3.0_ki*abb2(27)
      abb2(48)=abb2(39)*abb2(44)
      abb2(25)=2.0_ki*abb2(25)+abb2(48)
      abb2(25)=abb2(25)*abb2(38)
      abb2(48)=2.0_ki*spbl5k2
      abb2(19)=abb2(48)*abb2(19)
      abb2(49)=abb2(10)*spbl5k2
      abb2(50)=abb2(7)*abb2(49)
      abb2(50)=abb2(19)-3.0_ki*abb2(50)
      abb2(50)=abb2(7)*abb2(50)
      abb2(51)=-spbl5k2*abb2(16)
      abb2(50)=abb2(51)+abb2(50)
      abb2(50)=spbl4k2*abb2(50)
      abb2(39)=-abb2(6)*abb2(39)
      abb2(9)=abb2(9)*mT
      abb2(51)=spak1k2*abb2(9)
      abb2(40)=-mT*abb2(40)
      abb2(40)=abb2(40)+abb2(51)
      abb2(40)=abb2(40)*abb2(44)
      abb2(39)=abb2(39)+abb2(40)
      abb2(39)=abb2(39)*abb2(12)
      abb2(22)=-abb2(6)*abb2(15)*abb2(22)
      abb2(11)=-mT*abb2(11)
      abb2(9)=abb2(11)+abb2(9)
      abb2(9)=abb2(9)*abb2(44)
      abb2(9)=abb2(22)+abb2(9)
      abb2(9)=spbl4l3*abb2(9)
      abb2(11)=abb2(21)*spbl5k2
      abb2(21)=spbl5k1*abb2(11)
      abb2(9)=abb2(9)+abb2(21)
      abb2(9)=spak1l3*abb2(9)
      abb2(21)=abb2(5)*abb2(6)*abb2(42)
      abb2(9)=abb2(9)+abb2(39)+abb2(50)+abb2(21)+abb2(25)+abb2(18)+2.0_ki*abb2(&
      &37)+abb2(14)
      abb2(14)=2.0_ki*abb2(26)
      abb2(18)=-abb2(41)*abb2(10)
      abb2(10)=abb2(10)*abb2(43)
      abb2(21)=-abb2(7)*abb2(10)
      abb2(21)=abb2(18)+abb2(21)
      abb2(21)=abb2(21)*abb2(47)
      abb2(22)=abb2(18)*abb2(46)
      abb2(21)=abb2(21)+abb2(22)
      abb2(21)=abb2(8)*abb2(21)
      abb2(18)=abb2(6)*abb2(18)
      abb2(16)=abb2(16)*abb2(48)
      abb2(16)=abb2(18)+abb2(16)
      abb2(16)=abb2(5)*abb2(16)
      abb2(10)=-abb2(6)*abb2(10)
      abb2(10)=abb2(10)+abb2(19)
      abb2(10)=abb2(10)*abb2(38)
      abb2(18)=abb2(29)-2.0_ki*abb2(49)
      abb2(18)=abb2(18)*abb2(12)
      abb2(19)=-mT*abb2(23)
      abb2(22)=abb2(19)*abb2(27)
      abb2(25)=abb2(23)*spbl5k2
      abb2(27)=2.0_ki*abb2(25)-abb2(22)
      abb2(27)=spak1l3*spbl4l3*abb2(27)
      abb2(10)=abb2(21)+abb2(27)+abb2(18)+abb2(16)+abb2(10)-abb2(35)
      abb2(10)=2.0_ki*abb2(10)
      abb2(16)=-abb2(26)+abb2(31)
      abb2(16)=2.0_ki*abb2(16)
      abb2(18)=2.0_ki*abb2(36)
      abb2(21)=spak2l3*abb2(45)
      abb2(26)=abb2(29)-abb2(28)
      abb2(26)=spbl4l3*abb2(26)
      abb2(15)=-abb2(15)*abb2(23)
      abb2(27)=abb2(15)*abb2(6)
      abb2(28)=abb2(22)+abb2(27)
      abb2(31)=-spbl4k2*abb2(28)
      abb2(20)=abb2(20)*abb2(19)
      abb2(17)=-abb2(15)*abb2(17)
      abb2(35)=abb2(25)*abb2(12)
      abb2(33)=-abb2(33)*abb2(27)
      abb2(17)=abb2(33)+abb2(35)+abb2(31)+abb2(17)-abb2(20)
      abb2(17)=spak1l5*abb2(17)
      abb2(31)=abb2(23)*abb2(7)
      abb2(33)=-abb2(31)+abb2(19)
      abb2(33)=spbl4k2*abb2(7)*abb2(33)
      abb2(35)=abb2(19)*abb2(38)
      abb2(37)=spbl3k2*spak2l3*abb2(35)
      abb2(33)=abb2(33)+abb2(37)
      abb2(37)=-spbl5k2*abb2(19)
      abb2(38)=abb2(7)*abb2(25)
      abb2(37)=abb2(37)+abb2(38)
      abb2(37)=abb2(7)*abb2(37)
      abb2(20)=-spak2l3*abb2(20)
      abb2(13)=2.0_ki*abb2(13)
      abb2(13)=spbl5k1*abb2(13)
      abb2(27)=2.0_ki*abb2(27)
      abb2(22)=4.0_ki*abb2(22)-abb2(27)-abb2(25)
      abb2(22)=spbl4l3*abb2(22)
      abb2(25)=abb2(15)*abb2(5)
      abb2(38)=-abb2(25)+2.0_ki*abb2(35)
      abb2(38)=spbl5l3*abb2(38)
      abb2(22)=2.0_ki*abb2(38)+abb2(22)
      abb2(22)=spak2l3*abb2(22)
      abb2(30)=-spbl4k1*abb2(30)
      abb2(19)=-2.0_ki*abb2(31)+abb2(19)
      abb2(19)=abb2(7)*abb2(19)
      abb2(15)=abb2(19)-abb2(15)
      abb2(15)=spbl5l4*abb2(15)
      abb2(19)=-abb2(32)*abb2(29)*spbl3k1
      abb2(13)=abb2(19)+2.0_ki*abb2(15)+abb2(30)+abb2(13)+abb2(22)
      abb2(15)=2.0_ki*abb2(24)
      abb2(12)=abb2(23)*abb2(12)
      abb2(12)=abb2(12)-abb2(25)-abb2(35)
      abb2(12)=2.0_ki*abb2(12)
      abb2(19)=-2.0_ki*abb2(28)
      abb2(22)=-abb2(32)*abb2(27)
      R2d2=-abb2(36)
      rat2 = rat2 + R2d2
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='2' value='", &
          & R2d2, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd2h13
