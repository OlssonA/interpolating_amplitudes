module     p2_gg_httbar_abbrevd32h4_qp
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_kinematics_qp, only: epstensor
   use p2_gg_httbar_globalsh4_qp
   implicit none
   private
   complex(ki), dimension(56), public :: abb32
   complex(ki), public :: R2d32
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
      abb32(1)=1.0_ki/(mH**2-es34+es51-es23)
      abb32(2)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb32(3)=NC**(-1)
      abb32(4)=spak2l3**(-1)
      abb32(5)=spbl3k2**(-1)
      abb32(6)=spak2l4**(-1)
      abb32(7)=spbl4k2**(-1)
      abb32(8)=spbl5k2**(-1)
      abb32(9)=sqrt(mT**2)
      abb32(10)=abb32(6)*spae2k2
      abb32(11)=abb32(10)*mT**2
      abb32(12)=abb32(11)*abb32(7)
      abb32(13)=abb32(12)*spbk2e1
      abb32(14)=spbk2e1*spae2k2
      abb32(13)=abb32(13)-abb32(14)
      abb32(15)=spbl4e1*spae2l4
      abb32(16)=abb32(15)+abb32(13)
      abb32(17)=i_*TR*c2*e*gHT*gs**4*abb32(3)*abb32(2)
      abb32(18)=abb32(17)*abb32(1)
      abb32(19)=spbl4e2*abb32(18)
      abb32(20)=abb32(19)*abb32(9)
      abb32(21)=abb32(19)*mT
      abb32(22)=abb32(20)+abb32(21)
      abb32(23)=abb32(22)*abb32(16)
      abb32(24)=abb32(17)*abb32(9)
      abb32(25)=mT*abb32(1)
      abb32(26)=abb32(24)*abb32(25)
      abb32(27)=abb32(9)**2
      abb32(28)=abb32(27)*abb32(17)
      abb32(29)=abb32(28)*abb32(1)
      abb32(26)=abb32(26)+abb32(29)
      abb32(29)=abb32(10)*mT
      abb32(26)=abb32(26)*abb32(29)
      abb32(30)=-spbe2e1*abb32(26)
      abb32(23)=abb32(30)+abb32(23)
      abb32(23)=spae1l5*abb32(23)
      abb32(14)=abb32(15)-abb32(14)
      abb32(15)=spbl3k2*abb32(8)
      abb32(30)=abb32(15)*abb32(21)
      abb32(31)=abb32(30)*abb32(14)
      abb32(32)=abb32(10)*abb32(7)*abb32(15)*mT**3
      abb32(33)=abb32(19)*abb32(32)
      abb32(34)=spbk2e1*abb32(33)
      abb32(35)=abb32(11)*abb32(15)
      abb32(24)=abb32(1)*abb32(35)*abb32(24)
      abb32(36)=-spbe2e1*abb32(24)
      abb32(31)=abb32(36)+abb32(34)+abb32(31)
      abb32(31)=spae1l3*abb32(31)
      abb32(34)=abb32(4)*mH**2*spak2l5*abb32(5)
      abb32(36)=abb32(34)*spbk2e1
      abb32(37)=abb32(36)*abb32(20)
      abb32(38)=abb32(27)*abb32(19)
      abb32(39)=abb32(21)*abb32(9)
      abb32(39)=abb32(39)+abb32(38)
      abb32(39)=abb32(39)*mT
      abb32(40)=abb32(39)*abb32(8)
      abb32(41)=abb32(40)*spbk2e1
      abb32(37)=abb32(37)+abb32(41)
      abb32(41)=-spae1e2*abb32(37)
      abb32(11)=abb32(8)*abb32(11)
      abb32(42)=abb32(11)*spbk2e1
      abb32(43)=-abb32(22)*abb32(42)
      abb32(44)=abb32(21)*abb32(10)
      abb32(45)=-abb32(36)*abb32(44)
      abb32(43)=abb32(43)+abb32(45)
      abb32(43)=spae1l4*abb32(43)
      abb32(45)=abb32(10)*spal3l5
      abb32(46)=abb32(45)*abb32(21)
      abb32(47)=-spae1l4*abb32(46)
      abb32(48)=abb32(20)*spal3l5
      abb32(49)=-spae1e2*abb32(48)
      abb32(47)=abb32(49)+abb32(47)
      abb32(47)=spbl3e1*abb32(47)
      abb32(23)=abb32(31)+abb32(47)+abb32(43)+abb32(41)+abb32(23)
      abb32(23)=1.0_ki/2.0_ki*abb32(23)
      abb32(27)=abb32(21)*abb32(27)
      abb32(15)=abb32(27)*abb32(15)
      abb32(14)=abb32(15)*abb32(14)
      abb32(31)=spbk2e1*abb32(38)*abb32(32)
      abb32(32)=abb32(9)**3
      abb32(38)=abb32(18)*abb32(32)
      abb32(41)=-spbe2e1*abb32(35)*abb32(38)
      abb32(14)=abb32(41)+abb32(31)+abb32(14)
      abb32(14)=spae1l3*abb32(14)
      abb32(31)=abb32(32)*abb32(19)
      abb32(41)=abb32(31)+abb32(27)
      abb32(43)=abb32(41)*spae1e2
      abb32(39)=abb32(39)*abb32(10)
      abb32(47)=abb32(39)*spae1l4
      abb32(43)=abb32(43)+abb32(47)
      abb32(47)=spak1l5*abb32(43)
      abb32(49)=abb32(15)*spae1e2
      abb32(35)=abb32(35)*abb32(20)
      abb32(50)=abb32(35)*spae1l4
      abb32(49)=abb32(49)+abb32(50)
      abb32(50)=spak1l3*abb32(49)
      abb32(47)=abb32(50)+abb32(47)
      abb32(47)=spbk1e1*abb32(47)
      abb32(50)=-abb32(41)*abb32(42)
      abb32(51)=-abb32(10)*abb32(27)*abb32(36)
      abb32(50)=abb32(50)+abb32(51)
      abb32(50)=spae1l4*abb32(50)
      abb32(28)=abb32(28)*abb32(25)
      abb32(38)=abb32(38)+abb32(28)
      abb32(51)=abb32(38)*abb32(11)
      abb32(52)=-spae1k1*abb32(51)
      abb32(10)=abb32(28)*abb32(10)
      abb32(28)=abb32(10)*spae1k1
      abb32(53)=-abb32(34)*abb32(28)
      abb32(52)=abb32(52)+abb32(53)
      abb32(52)=spbe2e1*abb32(52)
      abb32(20)=abb32(34)*abb32(20)
      abb32(20)=abb32(40)+abb32(20)
      abb32(40)=abb32(20)*spbl4e1
      abb32(53)=spae2l4*spae1k1
      abb32(54)=abb32(53)*abb32(40)
      abb32(52)=abb32(52)+abb32(54)
      abb32(52)=spbk2k1*abb32(52)
      abb32(54)=-spae1e2*spal3l5*abb32(31)
      abb32(27)=-spae1l4*abb32(27)*abb32(45)
      abb32(12)=abb32(12)-spae2k2
      abb32(45)=abb32(48)*spae1k1
      abb32(55)=-abb32(45)*abb32(12)*spbk2k1
      abb32(27)=abb32(55)+abb32(54)+abb32(27)
      abb32(27)=spbl3e1*abb32(27)
      abb32(13)=abb32(45)*abb32(13)
      abb32(45)=abb32(10)*spal3l5
      abb32(54)=abb32(45)*spbe2e1
      abb32(55)=-spae1k1*abb32(54)
      abb32(56)=abb32(48)*abb32(53)*spbl4e1
      abb32(13)=abb32(56)+abb32(55)+abb32(13)
      abb32(13)=spbl3k1*abb32(13)
      abb32(41)=abb32(41)*spae1l5
      abb32(55)=abb32(41)*abb32(16)
      abb32(17)=-abb32(32)*abb32(25)*abb32(17)
      abb32(25)=abb32(9)**4
      abb32(18)=-abb32(25)*abb32(18)
      abb32(17)=abb32(18)+abb32(17)
      abb32(17)=spae1l5*abb32(17)*abb32(29)*spbe2e1
      abb32(18)=-abb32(25)*abb32(19)
      abb32(19)=-abb32(32)*abb32(21)
      abb32(18)=abb32(18)+abb32(19)
      abb32(18)=spbk2e1*abb32(18)*abb32(8)*mT
      abb32(19)=-abb32(31)*abb32(36)
      abb32(18)=abb32(18)+abb32(19)
      abb32(18)=spae1e2*abb32(18)
      abb32(19)=abb32(38)*abb32(42)
      abb32(21)=abb32(45)*spbl3e1
      abb32(19)=abb32(21)+abb32(19)
      abb32(21)=spae1k1*abb32(19)
      abb32(25)=abb32(36)*abb32(28)
      abb32(21)=abb32(25)+abb32(21)
      abb32(21)=spbe2k1*abb32(21)
      abb32(15)=abb32(15)*spae1l3
      abb32(15)=abb32(15)+abb32(41)
      abb32(25)=spak1e2*spbk1e1
      abb32(28)=-abb32(15)*abb32(25)
      abb32(29)=abb32(48)*spbl3e1
      abb32(31)=abb32(29)+abb32(37)
      abb32(32)=abb32(53)*spbl4k1
      abb32(37)=-abb32(31)*abb32(32)
      abb32(38)=abb32(39)*spae1l5
      abb32(35)=abb32(35)*spae1l3
      abb32(35)=abb32(38)+abb32(35)
      abb32(38)=spak1l4*spbk1e1
      abb32(39)=-abb32(35)*abb32(38)
      abb32(13)=abb32(13)+abb32(39)+abb32(37)+abb32(28)+abb32(21)+abb32(14)+abb&
      &32(27)+abb32(52)+abb32(50)+abb32(18)+abb32(17)+abb32(55)+abb32(47)
      abb32(14)=-2.0_ki*abb32(15)
      abb32(15)=abb32(10)*abb32(36)
      abb32(15)=abb32(15)+abb32(19)
      abb32(15)=2.0_ki*abb32(15)
      abb32(17)=spbe2k1*spae1k1
      abb32(18)=-abb32(26)*abb32(17)
      abb32(19)=abb32(12)*abb32(22)
      abb32(21)=spbk2k1*spae1k1
      abb32(27)=abb32(19)*abb32(21)
      abb32(28)=abb32(22)*abb32(32)
      abb32(18)=abb32(28)+abb32(18)+2.0_ki*abb32(43)+abb32(27)
      abb32(27)=-2.0_ki*spae2l4*abb32(31)
      abb32(28)=spae2l4*abb32(22)
      abb32(31)=-2.0_ki*abb32(35)
      abb32(16)=abb32(48)*abb32(16)
      abb32(16)=-abb32(54)+abb32(16)
      abb32(35)=-spak1l4*abb32(46)
      abb32(36)=-spak1e2*abb32(48)
      abb32(35)=abb32(35)+abb32(36)
      abb32(35)=spbk1e1*abb32(35)
      abb32(16)=2.0_ki*abb32(16)+abb32(35)
      abb32(35)=abb32(30)*spae2k2
      abb32(33)=abb32(35)-abb32(33)
      abb32(21)=-abb32(33)*abb32(21)
      abb32(17)=-abb32(24)*abb32(17)
      abb32(32)=abb32(30)*abb32(32)
      abb32(17)=abb32(32)+abb32(17)+2.0_ki*abb32(49)+abb32(21)
      abb32(21)=spae2l4*abb32(30)
      abb32(10)=-abb32(34)*abb32(10)
      abb32(10)=-abb32(51)+abb32(10)
      abb32(10)=spbe2e1*abb32(10)
      abb32(30)=spae2l4*abb32(40)
      abb32(12)=-abb32(12)*abb32(29)
      abb32(10)=abb32(12)+abb32(10)+abb32(30)
      abb32(12)=-abb32(20)*abb32(25)
      abb32(11)=abb32(11)*abb32(22)
      abb32(22)=abb32(44)*abb32(34)
      abb32(11)=abb32(11)+abb32(22)
      abb32(22)=-abb32(11)*abb32(38)
      abb32(10)=abb32(22)+2.0_ki*abb32(10)+abb32(12)
      R2d32=abb32(23)
      rat2 = rat2 + R2d32
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='32' value='", &
          & R2d32, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd32h4_qp
