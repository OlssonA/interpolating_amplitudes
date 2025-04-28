module     p2_gg_httbar_abbrevd34h4
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_kinematics, only: epstensor
   use p2_gg_httbar_globalsh4
   implicit none
   private
   complex(ki), dimension(56), public :: abb34
   complex(ki), public :: R2d34
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p2_gg_httbar_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_model
      use p2_gg_httbar_color, only: TR
      use p2_gg_httbar_globalsl1, only: epspow
      implicit none
      abb34(1)=1.0_ki/(-mT**2+es34)
      abb34(2)=1.0_ki/(es34-es51-es12)
      abb34(3)=NC**(-1)
      abb34(4)=spak2l3**(-1)
      abb34(5)=spbl3k2**(-1)
      abb34(6)=spbl5k2**(-1)
      abb34(7)=spak2l4**(-1)
      abb34(8)=spak2l5**(-1)
      abb34(9)=sqrt(mT**2)
      abb34(10)=abb34(6)*spbk2e2
      abb34(11)=abb34(10)*mT**2
      abb34(12)=abb34(11)*abb34(8)
      abb34(13)=abb34(12)*spae1k2
      abb34(14)=spae1k2*spbk2e2
      abb34(13)=abb34(13)-abb34(14)
      abb34(15)=spae1l5*spbl5e2
      abb34(16)=abb34(15)+abb34(13)
      abb34(17)=i_*TR*c1*e*gHT*gs**4*abb34(3)*abb34(2)
      abb34(18)=abb34(17)*abb34(1)
      abb34(19)=spae2l5*abb34(18)
      abb34(20)=abb34(19)*abb34(9)
      abb34(21)=abb34(19)*mT
      abb34(22)=abb34(20)+abb34(21)
      abb34(23)=-abb34(22)*abb34(16)
      abb34(24)=abb34(17)*abb34(9)
      abb34(25)=mT*abb34(1)
      abb34(26)=abb34(24)*abb34(25)
      abb34(27)=abb34(9)**2
      abb34(28)=abb34(27)*abb34(17)
      abb34(29)=abb34(28)*abb34(1)
      abb34(26)=abb34(26)+abb34(29)
      abb34(29)=abb34(10)*mT
      abb34(26)=abb34(26)*abb34(29)
      abb34(30)=spae1e2*abb34(26)
      abb34(23)=abb34(30)+abb34(23)
      abb34(23)=spbl4e1*abb34(23)
      abb34(14)=abb34(15)-abb34(14)
      abb34(15)=spak2l3*abb34(7)
      abb34(30)=abb34(15)*abb34(21)
      abb34(31)=-abb34(30)*abb34(14)
      abb34(32)=abb34(10)*abb34(8)*abb34(15)*mT**3
      abb34(33)=abb34(19)*abb34(32)
      abb34(34)=-spae1k2*abb34(33)
      abb34(35)=abb34(11)*abb34(15)
      abb34(24)=abb34(1)*abb34(35)*abb34(24)
      abb34(36)=spae1e2*abb34(24)
      abb34(31)=abb34(36)+abb34(34)+abb34(31)
      abb34(31)=spbl3e1*abb34(31)
      abb34(34)=abb34(4)*mH**2*spbl4k2*abb34(5)
      abb34(36)=abb34(34)*spae1k2
      abb34(37)=abb34(36)*abb34(20)
      abb34(38)=abb34(27)*abb34(19)
      abb34(39)=abb34(21)*abb34(9)
      abb34(39)=abb34(39)+abb34(38)
      abb34(39)=abb34(39)*mT
      abb34(40)=abb34(39)*abb34(7)
      abb34(41)=abb34(40)*spae1k2
      abb34(37)=abb34(37)+abb34(41)
      abb34(41)=spbe2e1*abb34(37)
      abb34(11)=abb34(7)*abb34(11)
      abb34(42)=abb34(11)*spae1k2
      abb34(43)=abb34(22)*abb34(42)
      abb34(44)=abb34(21)*abb34(10)
      abb34(45)=abb34(36)*abb34(44)
      abb34(43)=abb34(43)+abb34(45)
      abb34(43)=spbl5e1*abb34(43)
      abb34(45)=abb34(10)*spbl4l3
      abb34(46)=abb34(45)*abb34(21)
      abb34(47)=spbl5e1*abb34(46)
      abb34(48)=abb34(20)*spbl4l3
      abb34(49)=spbe2e1*abb34(48)
      abb34(47)=abb34(49)+abb34(47)
      abb34(47)=spae1l3*abb34(47)
      abb34(23)=abb34(47)+abb34(31)+abb34(43)+abb34(41)+abb34(23)
      abb34(23)=1.0_ki/2.0_ki*abb34(23)
      abb34(27)=abb34(21)*abb34(27)
      abb34(15)=abb34(27)*abb34(15)
      abb34(14)=-abb34(15)*abb34(14)
      abb34(31)=-spae1k2*abb34(38)*abb34(32)
      abb34(32)=abb34(9)**3
      abb34(38)=abb34(18)*abb34(32)
      abb34(41)=spae1e2*abb34(35)*abb34(38)
      abb34(14)=abb34(41)+abb34(31)+abb34(14)
      abb34(14)=spbl3e1*abb34(14)
      abb34(31)=abb34(32)*abb34(19)
      abb34(41)=abb34(31)+abb34(27)
      abb34(43)=abb34(41)*spbe2e1
      abb34(39)=abb34(39)*abb34(10)
      abb34(47)=abb34(39)*spbl5e1
      abb34(43)=abb34(43)+abb34(47)
      abb34(47)=-spbl4k1*abb34(43)
      abb34(49)=abb34(15)*spbe2e1
      abb34(35)=abb34(35)*abb34(20)
      abb34(50)=abb34(35)*spbl5e1
      abb34(49)=abb34(49)+abb34(50)
      abb34(50)=-spbl3k1*abb34(49)
      abb34(47)=abb34(50)+abb34(47)
      abb34(47)=spae1k1*abb34(47)
      abb34(50)=abb34(41)*abb34(42)
      abb34(51)=abb34(10)*abb34(27)*abb34(36)
      abb34(50)=abb34(50)+abb34(51)
      abb34(50)=spbl5e1*abb34(50)
      abb34(28)=abb34(28)*abb34(25)
      abb34(38)=abb34(38)+abb34(28)
      abb34(51)=abb34(38)*abb34(11)
      abb34(52)=spbk1e1*abb34(51)
      abb34(10)=abb34(28)*abb34(10)
      abb34(28)=abb34(10)*spbk1e1
      abb34(53)=abb34(34)*abb34(28)
      abb34(52)=abb34(52)+abb34(53)
      abb34(52)=spae1e2*abb34(52)
      abb34(20)=abb34(34)*abb34(20)
      abb34(20)=abb34(40)+abb34(20)
      abb34(40)=abb34(20)*spae1l5
      abb34(53)=spbl5e2*spbk1e1
      abb34(54)=-abb34(53)*abb34(40)
      abb34(52)=abb34(52)+abb34(54)
      abb34(52)=spak1k2*abb34(52)
      abb34(54)=spbe2e1*spbl4l3*abb34(31)
      abb34(27)=spbl5e1*abb34(27)*abb34(45)
      abb34(12)=abb34(12)-spbk2e2
      abb34(45)=abb34(48)*spbk1e1
      abb34(55)=abb34(45)*abb34(12)*spak1k2
      abb34(27)=abb34(55)+abb34(54)+abb34(27)
      abb34(27)=spae1l3*abb34(27)
      abb34(13)=-abb34(45)*abb34(13)
      abb34(45)=abb34(10)*spbl4l3
      abb34(54)=abb34(45)*spae1e2
      abb34(55)=spbk1e1*abb34(54)
      abb34(56)=-abb34(48)*abb34(53)*spae1l5
      abb34(13)=abb34(56)+abb34(55)+abb34(13)
      abb34(13)=spak1l3*abb34(13)
      abb34(41)=abb34(41)*spbl4e1
      abb34(55)=-abb34(41)*abb34(16)
      abb34(56)=abb34(9)**4
      abb34(19)=abb34(56)*abb34(19)
      abb34(21)=abb34(32)*abb34(21)
      abb34(19)=abb34(19)+abb34(21)
      abb34(19)=spae1k2*abb34(19)*abb34(7)*mT
      abb34(21)=abb34(31)*abb34(36)
      abb34(19)=abb34(19)+abb34(21)
      abb34(19)=spbe2e1*abb34(19)
      abb34(17)=abb34(32)*abb34(25)*abb34(17)
      abb34(18)=abb34(56)*abb34(18)
      abb34(17)=abb34(18)+abb34(17)
      abb34(17)=spbl4e1*abb34(17)*abb34(29)*spae1e2
      abb34(15)=abb34(15)*spbl3e1
      abb34(15)=abb34(15)+abb34(41)
      abb34(18)=spbe2k1*spae1k1
      abb34(21)=abb34(15)*abb34(18)
      abb34(25)=abb34(38)*abb34(42)
      abb34(29)=abb34(45)*spae1l3
      abb34(25)=abb34(29)+abb34(25)
      abb34(29)=-spbk1e1*abb34(25)
      abb34(28)=-abb34(36)*abb34(28)
      abb34(28)=abb34(28)+abb34(29)
      abb34(28)=spak1e2*abb34(28)
      abb34(29)=abb34(39)*spbl4e1
      abb34(31)=abb34(35)*spbl3e1
      abb34(29)=abb34(29)+abb34(31)
      abb34(31)=spbl5k1*spae1k1
      abb34(32)=abb34(29)*abb34(31)
      abb34(35)=abb34(48)*spae1l3
      abb34(37)=abb34(35)+abb34(37)
      abb34(38)=abb34(53)*spak1l5
      abb34(39)=abb34(37)*abb34(38)
      abb34(13)=abb34(13)+abb34(39)+abb34(32)+abb34(28)+abb34(21)+abb34(27)+abb&
      &34(14)+abb34(52)+abb34(50)+abb34(17)+abb34(19)+abb34(55)+abb34(47)
      abb34(14)=-abb34(10)*abb34(36)
      abb34(14)=abb34(14)-abb34(25)
      abb34(14)=2.0_ki*abb34(14)
      abb34(15)=2.0_ki*abb34(15)
      abb34(17)=2.0_ki*abb34(29)
      abb34(19)=2.0_ki*spbl5e2*abb34(37)
      abb34(21)=spak1e2*spbk1e1
      abb34(25)=abb34(26)*abb34(21)
      abb34(27)=abb34(12)*abb34(22)
      abb34(28)=spak1k2*spbk1e1
      abb34(29)=-abb34(27)*abb34(28)
      abb34(32)=-abb34(22)*abb34(38)
      abb34(25)=abb34(32)+abb34(25)-2.0_ki*abb34(43)+abb34(29)
      abb34(29)=-spbl5e2*abb34(22)
      abb34(32)=abb34(30)*spbk2e2
      abb34(32)=abb34(32)-abb34(33)
      abb34(28)=abb34(32)*abb34(28)
      abb34(21)=abb34(24)*abb34(21)
      abb34(33)=-abb34(30)*abb34(38)
      abb34(21)=abb34(33)+abb34(21)-2.0_ki*abb34(49)+abb34(28)
      abb34(28)=-spbl5e2*abb34(30)
      abb34(16)=-abb34(48)*abb34(16)
      abb34(16)=abb34(54)+abb34(16)
      abb34(30)=spbl5k1*abb34(46)
      abb34(33)=spbe2k1*abb34(48)
      abb34(30)=abb34(30)+abb34(33)
      abb34(30)=spae1k1*abb34(30)
      abb34(16)=2.0_ki*abb34(16)+abb34(30)
      abb34(10)=abb34(34)*abb34(10)
      abb34(10)=abb34(51)+abb34(10)
      abb34(10)=spae1e2*abb34(10)
      abb34(30)=-spbl5e2*abb34(40)
      abb34(12)=abb34(12)*abb34(35)
      abb34(10)=abb34(12)+abb34(10)+abb34(30)
      abb34(12)=abb34(20)*abb34(18)
      abb34(11)=abb34(11)*abb34(22)
      abb34(18)=abb34(44)*abb34(34)
      abb34(11)=abb34(11)+abb34(18)
      abb34(18)=abb34(11)*abb34(31)
      abb34(10)=abb34(18)+2.0_ki*abb34(10)+abb34(12)
      R2d34=abb34(23)
      rat2 = rat2 + R2d34
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='34' value='", &
          & R2d34, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd34h4
