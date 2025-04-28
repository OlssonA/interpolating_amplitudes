module     p2_gg_httbar_abbrevd89h4
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_kinematics, only: epstensor
   use p2_gg_httbar_globalsh4
   implicit none
   private
   complex(ki), dimension(57), public :: abb89
   complex(ki), public :: R2d89
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
      abb89(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb89(2)=sqrt(mT**2)
      abb89(3)=NC**(-1)
      abb89(4)=spak2l4**(-1)
      abb89(5)=spbl5k2**(-1)
      abb89(6)=spak2l3**(-1)
      abb89(7)=spbl3k2**(-1)
      abb89(8)=abb89(2)**2
      abb89(9)=i_*TR*e*gHT*abb89(1)*gs**4
      abb89(10)=abb89(8)*abb89(9)
      abb89(11)=abb89(9)*abb89(2)
      abb89(12)=abb89(11)*mT
      abb89(10)=abb89(12)+abb89(10)
      abb89(12)=c2*abb89(3)
      abb89(12)=abb89(12)-c3
      abb89(13)=-mT*abb89(12)
      abb89(10)=-abb89(10)*abb89(13)
      abb89(14)=abb89(10)*spae1k2
      abb89(15)=abb89(14)*spak1l5
      abb89(16)=-abb89(11)*abb89(12)
      abb89(17)=mT**2
      abb89(18)=-abb89(17)*abb89(16)
      abb89(19)=spbl3k2*abb89(5)
      abb89(20)=abb89(19)*spae1k2
      abb89(21)=abb89(20)*abb89(18)
      abb89(22)=abb89(21)*spak1l3
      abb89(15)=abb89(15)+abb89(22)
      abb89(23)=spbk2k1*abb89(15)
      abb89(24)=abb89(9)*mT
      abb89(8)=abb89(24)*abb89(8)
      abb89(25)=-abb89(8)*abb89(12)
      abb89(26)=spal3l5*spae1k2
      abb89(27)=-spbl3k2*abb89(25)*abb89(26)
      abb89(28)=abb89(2)**3
      abb89(29)=abb89(28)*abb89(24)
      abb89(30)=abb89(9)*abb89(2)**4
      abb89(29)=abb89(30)+abb89(29)
      abb89(13)=-abb89(29)*abb89(13)
      abb89(29)=spae1l5*abb89(13)
      abb89(9)=abb89(28)*abb89(9)
      abb89(28)=abb89(12)*abb89(9)
      abb89(30)=abb89(28)*abb89(17)
      abb89(31)=abb89(19)*spae1l3
      abb89(32)=abb89(30)*abb89(31)
      abb89(23)=abb89(32)+abb89(29)+abb89(27)+abb89(23)
      abb89(27)=spbe2e1*abb89(4)
      abb89(29)=abb89(27)*spae2k2
      abb89(23)=abb89(29)*abb89(23)
      abb89(32)=spak2l5*mH**2*abb89(7)*abb89(6)
      abb89(33)=abb89(28)*abb89(32)
      abb89(34)=abb89(13)*abb89(5)
      abb89(33)=abb89(33)+abb89(34)
      abb89(8)=abb89(9)+abb89(8)
      abb89(8)=-abb89(8)*abb89(12)
      abb89(9)=spak2l5*abb89(8)
      abb89(34)=abb89(25)*abb89(19)
      abb89(35)=spak2l3*abb89(34)
      abb89(36)=abb89(16)*spal3l5
      abb89(37)=abb89(36)*spbl3k1
      abb89(38)=-spak1k2*abb89(37)
      abb89(9)=abb89(38)+abb89(35)+abb89(9)-abb89(33)
      abb89(35)=spbk2e2*spae1e2
      abb89(38)=abb89(35)*spbl4e1
      abb89(9)=abb89(38)*abb89(9)
      abb89(39)=abb89(32)*abb89(16)
      abb89(40)=abb89(10)*abb89(5)
      abb89(39)=abb89(39)-abb89(40)
      abb89(41)=abb89(38)*abb89(39)
      abb89(42)=abb89(10)*spae1l5
      abb89(18)=abb89(31)*abb89(18)
      abb89(18)=abb89(42)+abb89(18)
      abb89(31)=abb89(29)*abb89(18)
      abb89(31)=abb89(31)+abb89(41)
      abb89(41)=-es12*abb89(31)
      abb89(42)=-spbk2e1*abb89(39)
      abb89(43)=abb89(35)*spak1k2*spbl4k1
      abb89(44)=-abb89(42)*abb89(43)
      abb89(17)=abb89(17)*abb89(5)
      abb89(45)=-abb89(8)*abb89(17)
      abb89(46)=abb89(45)*abb89(4)
      abb89(47)=abb89(32)*abb89(4)
      abb89(48)=abb89(47)*abb89(25)
      abb89(46)=abb89(48)-abb89(46)
      abb89(46)=abb89(35)*abb89(46)
      abb89(48)=spbk1e1*spak1k2
      abb89(49)=-abb89(46)*abb89(48)
      abb89(50)=abb89(8)*spae2l5
      abb89(51)=abb89(50)*spbe2e1
      abb89(52)=spae1k1*spbl4k1
      abb89(53)=abb89(52)*abb89(51)
      abb89(54)=spbl4e2*spae1e2
      abb89(33)=abb89(54)*spbk2e1*abb89(33)
      abb89(43)=abb89(36)*abb89(43)
      abb89(28)=abb89(28)*spal3l5
      abb89(54)=abb89(28)*abb89(54)
      abb89(43)=abb89(43)+abb89(54)
      abb89(43)=spbl3e1*abb89(43)
      abb89(54)=spbl4e1*spae1e2
      abb89(28)=-abb89(54)*abb89(28)
      abb89(55)=spal3l5*abb89(4)
      abb89(56)=abb89(55)*abb89(25)
      abb89(57)=-abb89(48)*abb89(56)*spae1e2
      abb89(28)=abb89(28)+abb89(57)
      abb89(28)=spbl3e2*abb89(28)
      abb89(20)=-abb89(30)*abb89(27)*abb89(20)
      abb89(30)=abb89(52)*abb89(34)*spbe2e1
      abb89(20)=abb89(20)+abb89(30)
      abb89(20)=spae2l3*abb89(20)
      abb89(13)=-spae2l5*spae1k2*abb89(13)*abb89(27)
      abb89(9)=abb89(20)+abb89(28)+abb89(41)+abb89(43)+abb89(33)+abb89(53)+abb8&
      &9(49)+abb89(13)+abb89(44)+abb89(9)+abb89(23)
      abb89(13)=-2.0_ki*abb89(31)
      abb89(20)=abb89(36)*spbl3e1
      abb89(23)=abb89(42)-abb89(20)
      abb89(28)=abb89(23)*spae1e2
      abb89(30)=spbl4e2*abb89(28)
      abb89(33)=-spae2l5*abb89(14)
      abb89(41)=-spae2l3*abb89(21)
      abb89(33)=abb89(41)+abb89(33)
      abb89(33)=abb89(27)*abb89(33)
      abb89(41)=spbl3e2*abb89(54)*abb89(36)
      abb89(30)=abb89(41)+abb89(33)+abb89(30)+abb89(31)
      abb89(31)=abb89(4)*abb89(18)
      abb89(33)=-spak1k2*abb89(31)
      abb89(41)=spae1k2*abb89(4)
      abb89(10)=abb89(41)*abb89(10)
      abb89(42)=spak1l5*abb89(10)
      abb89(22)=abb89(4)*abb89(22)
      abb89(22)=abb89(22)+abb89(42)+abb89(33)
      abb89(22)=spbe2k1*abb89(22)
      abb89(33)=-abb89(25)*abb89(32)
      abb89(33)=abb89(45)+abb89(33)
      abb89(33)=abb89(41)*abb89(33)
      abb89(41)=-abb89(24)*abb89(12)
      abb89(19)=abb89(41)*abb89(19)
      abb89(42)=abb89(19)*spak2l3
      abb89(11)=abb89(24)+abb89(11)
      abb89(11)=-abb89(11)*abb89(12)
      abb89(12)=abb89(11)*spak2l5
      abb89(12)=abb89(42)+abb89(12)
      abb89(24)=-abb89(52)*abb89(12)
      abb89(24)=abb89(24)+abb89(33)
      abb89(24)=spbk2e2*abb89(24)
      abb89(33)=abb89(56)*spbl3e2
      abb89(42)=-spae1k2*abb89(33)
      abb89(22)=abb89(42)+abb89(24)+abb89(22)
      abb89(24)=abb89(34)*spae2l3
      abb89(42)=-spak1e2*abb89(37)
      abb89(42)=abb89(42)+abb89(24)+abb89(50)
      abb89(42)=spbl4e1*abb89(42)
      abb89(43)=-spbl4k1*abb89(23)
      abb89(44)=abb89(39)*spbl4e1
      abb89(49)=-spbk2k1*abb89(44)
      abb89(43)=abb89(49)+abb89(43)
      abb89(43)=spak1e2*abb89(43)
      abb89(49)=abb89(55)*abb89(41)
      abb89(50)=abb89(49)*abb89(48)
      abb89(53)=spae2k2*spbl3k2
      abb89(55)=abb89(50)*abb89(53)
      abb89(42)=abb89(43)+abb89(55)+abb89(42)
      abb89(8)=abb89(8)*abb89(54)
      abb89(43)=-abb89(11)*abb89(52)
      abb89(10)=2.0_ki*abb89(10)+abb89(43)
      abb89(20)=-spae1k2*spbk2e2*abb89(20)
      abb89(43)=-2.0_ki*abb89(23)
      abb89(24)=-spbe2e1*abb89(24)
      abb89(24)=-abb89(51)+abb89(24)
      abb89(12)=spbk2e2*abb89(12)
      abb89(51)=spbl4e1*spbk2e2
      abb89(16)=abb89(51)*abb89(16)
      abb89(25)=abb89(25)*abb89(27)
      abb89(16)=abb89(16)-abb89(25)
      abb89(25)=abb89(16)*abb89(26)
      abb89(26)=spbl4e1*abb89(36)
      abb89(26)=-2.0_ki*abb89(26)+abb89(50)
      abb89(34)=abb89(54)*abb89(34)
      abb89(50)=abb89(4)*abb89(21)
      abb89(52)=-abb89(52)*abb89(19)
      abb89(50)=2.0_ki*abb89(50)+abb89(52)
      abb89(45)=abb89(27)*abb89(45)
      abb89(16)=abb89(16)*abb89(32)
      abb89(32)=-abb89(51)*abb89(40)
      abb89(16)=abb89(16)+abb89(45)+abb89(32)
      abb89(16)=spae1k2*abb89(16)
      abb89(17)=abb89(4)*abb89(11)*abb89(17)
      abb89(32)=abb89(47)*abb89(41)
      abb89(17)=abb89(17)+abb89(32)
      abb89(32)=abb89(17)*abb89(48)
      abb89(32)=-2.0_ki*abb89(44)+abb89(32)
      abb89(40)=spbk2e1*spae2k2*abb89(31)
      abb89(31)=-2.0_ki*abb89(31)
      abb89(33)=spae1e2*abb89(33)
      abb89(33)=abb89(33)+abb89(46)
      abb89(41)=-abb89(49)*abb89(53)
      abb89(18)=spak1k2*abb89(18)
      abb89(15)=abb89(18)-abb89(15)
      abb89(15)=abb89(27)*abb89(15)
      abb89(18)=spbk2k1*abb89(39)
      abb89(18)=abb89(37)+abb89(18)
      abb89(18)=abb89(54)*abb89(18)
      abb89(27)=spbl4k1*abb89(28)
      abb89(18)=abb89(27)+abb89(18)
      abb89(14)=abb89(29)*abb89(14)
      abb89(21)=abb89(29)*abb89(21)
      abb89(23)=-abb89(35)*abb89(23)
      abb89(27)=-abb89(36)*abb89(38)
      R2d89=0.0_ki
      rat2 = rat2 + R2d89
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='89' value='", &
          & R2d89, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd89h4
