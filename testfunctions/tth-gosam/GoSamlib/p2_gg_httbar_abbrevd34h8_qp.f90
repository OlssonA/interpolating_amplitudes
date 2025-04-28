module     p2_gg_httbar_abbrevd34h8_qp
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_kinematics_qp, only: epstensor
   use p2_gg_httbar_globalsh8_qp
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
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_color_qp, only: TR
      use p2_gg_httbar_globalsl1_qp, only: epspow
      implicit none
      abb34(1)=1.0_ki/(-mT**2+es34)
      abb34(2)=1.0_ki/(es34-es51-es12)
      abb34(3)=NC**(-1)
      abb34(4)=spak2l3**(-1)
      abb34(5)=spbl3k2**(-1)
      abb34(6)=spak2l5**(-1)
      abb34(7)=spbl4k2**(-1)
      abb34(8)=spbl5k2**(-1)
      abb34(9)=sqrt(mT**2)
      abb34(10)=abb34(6)*spae2k2
      abb34(11)=abb34(10)*mT**2
      abb34(12)=abb34(11)*abb34(8)
      abb34(13)=abb34(12)*spbk2e1
      abb34(14)=spbk2e1*spae2k2
      abb34(13)=abb34(13)-abb34(14)
      abb34(15)=spbl5e1*spae2l5
      abb34(16)=abb34(15)+abb34(13)
      abb34(17)=i_*TR*c1*e*gHT*gs**4*abb34(3)*abb34(2)
      abb34(18)=abb34(17)*abb34(1)
      abb34(19)=spbl5e2*abb34(18)
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
      abb34(30)=spbe2e1*abb34(26)
      abb34(23)=abb34(30)+abb34(23)
      abb34(23)=spae1l4*abb34(23)
      abb34(14)=abb34(15)-abb34(14)
      abb34(15)=spbl3k2*abb34(7)
      abb34(30)=abb34(15)*abb34(21)
      abb34(31)=-abb34(30)*abb34(14)
      abb34(32)=abb34(10)*abb34(8)*abb34(15)*mT**3
      abb34(33)=abb34(19)*abb34(32)
      abb34(34)=-spbk2e1*abb34(33)
      abb34(35)=abb34(11)*abb34(15)
      abb34(24)=abb34(1)*abb34(35)*abb34(24)
      abb34(36)=spbe2e1*abb34(24)
      abb34(31)=abb34(36)+abb34(34)+abb34(31)
      abb34(31)=spae1l3*abb34(31)
      abb34(34)=abb34(4)*mH**2*spak2l4*abb34(5)
      abb34(36)=abb34(34)*spbk2e1
      abb34(37)=abb34(36)*abb34(20)
      abb34(38)=abb34(27)*abb34(19)
      abb34(39)=abb34(21)*abb34(9)
      abb34(39)=abb34(39)+abb34(38)
      abb34(39)=abb34(39)*mT
      abb34(40)=abb34(39)*abb34(7)
      abb34(41)=abb34(40)*spbk2e1
      abb34(37)=abb34(37)+abb34(41)
      abb34(41)=spae1e2*abb34(37)
      abb34(11)=abb34(7)*abb34(11)
      abb34(42)=abb34(11)*spbk2e1
      abb34(43)=abb34(22)*abb34(42)
      abb34(44)=abb34(21)*abb34(10)
      abb34(45)=abb34(36)*abb34(44)
      abb34(43)=abb34(43)+abb34(45)
      abb34(43)=spae1l5*abb34(43)
      abb34(45)=abb34(10)*spal3l4
      abb34(46)=abb34(45)*abb34(21)
      abb34(47)=spae1l5*abb34(46)
      abb34(48)=abb34(20)*spal3l4
      abb34(49)=spae1e2*abb34(48)
      abb34(47)=abb34(49)+abb34(47)
      abb34(47)=spbl3e1*abb34(47)
      abb34(23)=abb34(31)+abb34(47)+abb34(43)+abb34(41)+abb34(23)
      abb34(23)=1.0_ki/2.0_ki*abb34(23)
      abb34(27)=abb34(21)*abb34(27)
      abb34(15)=abb34(27)*abb34(15)
      abb34(14)=-abb34(15)*abb34(14)
      abb34(31)=-spbk2e1*abb34(38)*abb34(32)
      abb34(32)=abb34(9)**3
      abb34(38)=abb34(18)*abb34(32)
      abb34(41)=spbe2e1*abb34(35)*abb34(38)
      abb34(14)=abb34(41)+abb34(31)+abb34(14)
      abb34(14)=spae1l3*abb34(14)
      abb34(31)=abb34(32)*abb34(19)
      abb34(41)=abb34(31)+abb34(27)
      abb34(43)=abb34(41)*spae1e2
      abb34(39)=abb34(39)*abb34(10)
      abb34(47)=abb34(39)*spae1l5
      abb34(43)=abb34(43)+abb34(47)
      abb34(47)=-spak1l4*abb34(43)
      abb34(49)=abb34(15)*spae1e2
      abb34(35)=abb34(35)*abb34(20)
      abb34(50)=abb34(35)*spae1l5
      abb34(49)=abb34(49)+abb34(50)
      abb34(50)=-spak1l3*abb34(49)
      abb34(47)=abb34(50)+abb34(47)
      abb34(47)=spbk1e1*abb34(47)
      abb34(50)=abb34(41)*abb34(42)
      abb34(51)=abb34(10)*abb34(27)*abb34(36)
      abb34(50)=abb34(50)+abb34(51)
      abb34(50)=spae1l5*abb34(50)
      abb34(28)=abb34(28)*abb34(25)
      abb34(38)=abb34(38)+abb34(28)
      abb34(51)=abb34(38)*abb34(11)
      abb34(52)=spae1k1*abb34(51)
      abb34(10)=abb34(28)*abb34(10)
      abb34(28)=abb34(10)*spae1k1
      abb34(53)=abb34(34)*abb34(28)
      abb34(52)=abb34(52)+abb34(53)
      abb34(52)=spbe2e1*abb34(52)
      abb34(20)=abb34(34)*abb34(20)
      abb34(20)=abb34(40)+abb34(20)
      abb34(40)=abb34(20)*spbl5e1
      abb34(53)=spae2l5*spae1k1
      abb34(54)=-abb34(53)*abb34(40)
      abb34(52)=abb34(52)+abb34(54)
      abb34(52)=spbk2k1*abb34(52)
      abb34(54)=spae1e2*spal3l4*abb34(31)
      abb34(27)=spae1l5*abb34(27)*abb34(45)
      abb34(12)=abb34(12)-spae2k2
      abb34(45)=abb34(48)*spae1k1
      abb34(55)=abb34(45)*abb34(12)*spbk2k1
      abb34(27)=abb34(55)+abb34(54)+abb34(27)
      abb34(27)=spbl3e1*abb34(27)
      abb34(13)=-abb34(45)*abb34(13)
      abb34(45)=abb34(10)*spal3l4
      abb34(54)=abb34(45)*spbe2e1
      abb34(55)=spae1k1*abb34(54)
      abb34(56)=-abb34(48)*abb34(53)*spbl5e1
      abb34(13)=abb34(56)+abb34(55)+abb34(13)
      abb34(13)=spbl3k1*abb34(13)
      abb34(41)=abb34(41)*spae1l4
      abb34(55)=-abb34(41)*abb34(16)
      abb34(17)=abb34(32)*abb34(25)*abb34(17)
      abb34(25)=abb34(9)**4
      abb34(18)=abb34(25)*abb34(18)
      abb34(17)=abb34(18)+abb34(17)
      abb34(17)=spae1l4*abb34(17)*abb34(29)*spbe2e1
      abb34(18)=abb34(25)*abb34(19)
      abb34(19)=abb34(32)*abb34(21)
      abb34(18)=abb34(18)+abb34(19)
      abb34(18)=spbk2e1*abb34(18)*abb34(7)*mT
      abb34(19)=abb34(31)*abb34(36)
      abb34(18)=abb34(18)+abb34(19)
      abb34(18)=spae1e2*abb34(18)
      abb34(19)=abb34(38)*abb34(42)
      abb34(21)=abb34(45)*spbl3e1
      abb34(19)=abb34(21)+abb34(19)
      abb34(21)=-spae1k1*abb34(19)
      abb34(25)=-abb34(36)*abb34(28)
      abb34(21)=abb34(25)+abb34(21)
      abb34(21)=spbe2k1*abb34(21)
      abb34(15)=abb34(15)*spae1l3
      abb34(15)=abb34(15)+abb34(41)
      abb34(25)=spak1e2*spbk1e1
      abb34(28)=abb34(15)*abb34(25)
      abb34(29)=abb34(48)*spbl3e1
      abb34(31)=abb34(29)+abb34(37)
      abb34(32)=abb34(53)*spbl5k1
      abb34(37)=abb34(31)*abb34(32)
      abb34(38)=abb34(39)*spae1l4
      abb34(35)=abb34(35)*spae1l3
      abb34(35)=abb34(38)+abb34(35)
      abb34(38)=spak1l5*spbk1e1
      abb34(39)=abb34(35)*abb34(38)
      abb34(13)=abb34(13)+abb34(39)+abb34(37)+abb34(28)+abb34(21)+abb34(14)+abb&
      &34(27)+abb34(52)+abb34(50)+abb34(18)+abb34(17)+abb34(55)+abb34(47)
      abb34(14)=2.0_ki*abb34(15)
      abb34(15)=-abb34(10)*abb34(36)
      abb34(15)=abb34(15)-abb34(19)
      abb34(15)=2.0_ki*abb34(15)
      abb34(17)=2.0_ki*spae2l5*abb34(31)
      abb34(18)=2.0_ki*abb34(35)
      abb34(19)=spbe2k1*spae1k1
      abb34(21)=abb34(26)*abb34(19)
      abb34(27)=abb34(12)*abb34(22)
      abb34(28)=spbk2k1*spae1k1
      abb34(31)=-abb34(27)*abb34(28)
      abb34(35)=-abb34(22)*abb34(32)
      abb34(21)=abb34(35)+abb34(21)-2.0_ki*abb34(43)+abb34(31)
      abb34(31)=-spae2l5*abb34(22)
      abb34(16)=-abb34(48)*abb34(16)
      abb34(16)=abb34(54)+abb34(16)
      abb34(35)=spak1l5*abb34(46)
      abb34(36)=spak1e2*abb34(48)
      abb34(35)=abb34(35)+abb34(36)
      abb34(35)=spbk1e1*abb34(35)
      abb34(16)=2.0_ki*abb34(16)+abb34(35)
      abb34(35)=abb34(30)*spae2k2
      abb34(33)=abb34(35)-abb34(33)
      abb34(28)=abb34(33)*abb34(28)
      abb34(19)=abb34(24)*abb34(19)
      abb34(32)=-abb34(30)*abb34(32)
      abb34(19)=abb34(32)+abb34(19)-2.0_ki*abb34(49)+abb34(28)
      abb34(28)=-spae2l5*abb34(30)
      abb34(10)=abb34(34)*abb34(10)
      abb34(10)=abb34(51)+abb34(10)
      abb34(10)=spbe2e1*abb34(10)
      abb34(30)=-spae2l5*abb34(40)
      abb34(12)=abb34(12)*abb34(29)
      abb34(10)=abb34(12)+abb34(10)+abb34(30)
      abb34(12)=abb34(20)*abb34(25)
      abb34(11)=abb34(11)*abb34(22)
      abb34(22)=abb34(44)*abb34(34)
      abb34(11)=abb34(11)+abb34(22)
      abb34(22)=abb34(11)*abb34(38)
      abb34(10)=abb34(22)+2.0_ki*abb34(10)+abb34(12)
      R2d34=abb34(23)
      rat2 = rat2 + R2d34
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='34' value='", &
          & R2d34, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd34h8_qp
