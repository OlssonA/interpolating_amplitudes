module     p2_gg_httbar_abbrevd254h0
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_kinematics, only: epstensor
   use p2_gg_httbar_globalsh0
   implicit none
   private
   complex(ki), dimension(60), public :: abb254
   complex(ki), public :: R2d254
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
      abb254(1)=sqrt(mT**2)
      abb254(2)=NC**(-1)
      abb254(3)=spbl5k2**(-1)
      abb254(4)=spak2l3**(-1)
      abb254(5)=spbl3k2**(-1)
      abb254(6)=spbl4k2**(-1)
      abb254(7)=c1*abb254(2)
      abb254(7)=abb254(7)-c3
      abb254(7)=abb254(7)*gs**4*i_*TR*e*gHT
      abb254(8)=-abb254(7)*abb254(1)**3
      abb254(9)=abb254(3)*mT
      abb254(10)=abb254(8)*abb254(9)
      abb254(11)=spbe2e1*spae2l4
      abb254(12)=abb254(10)*abb254(11)
      abb254(13)=-abb254(7)*abb254(1)**2
      abb254(14)=abb254(11)*abb254(13)
      abb254(15)=abb254(14)*spak2l5
      abb254(16)=abb254(13)*spae2l5
      abb254(17)=abb254(16)*spbe2e1
      abb254(18)=abb254(17)*spak2l4
      abb254(12)=-abb254(15)+abb254(12)+abb254(18)
      abb254(15)=spae1l3*spbl3k2
      abb254(18)=spbk2k1*spae1k1
      abb254(18)=abb254(15)+2.0_ki*abb254(18)
      abb254(18)=abb254(12)*abb254(18)
      abb254(19)=spak1l5*spbk1e1
      abb254(20)=spbl3e1*spal3l5
      abb254(19)=-abb254(20)+2.0_ki*abb254(19)
      abb254(20)=abb254(6)*mT
      abb254(21)=abb254(20)*spbk2e2
      abb254(22)=abb254(8)*abb254(21)
      abb254(23)=abb254(22)*spae1e2
      abb254(24)=abb254(23)*abb254(19)
      abb254(25)=abb254(9)*spae1e2
      abb254(26)=abb254(8)*abb254(25)
      abb254(27)=spak2l4*spbk2e2
      abb254(28)=abb254(27)*abb254(26)
      abb254(29)=mH**2*abb254(5)*abb254(4)
      abb254(30)=abb254(29)*spak2l5
      abb254(31)=-abb254(30)*abb254(23)
      abb254(28)=abb254(28)+abb254(31)
      abb254(28)=spbk2e1*abb254(28)
      abb254(31)=spbk1e1*spae2l4
      abb254(32)=abb254(31)*spae1l5
      abb254(33)=abb254(32)*abb254(13)
      abb254(34)=spak1l3*abb254(33)
      abb254(35)=spak1l3*spbk1e1
      abb254(36)=abb254(16)*spae1l4
      abb254(37)=-abb254(36)*abb254(35)
      abb254(34)=abb254(34)+abb254(37)
      abb254(34)=spbl3e2*abb254(34)
      abb254(37)=abb254(29)*abb254(13)
      abb254(38)=abb254(37)*spbk2e2
      abb254(32)=abb254(38)*abb254(32)
      abb254(39)=abb254(29)*spbk2e2
      abb254(40)=abb254(36)*spbk1e1
      abb254(41)=-abb254(40)*abb254(39)
      abb254(32)=abb254(32)+abb254(41)
      abb254(32)=spak1k2*abb254(32)
      abb254(41)=-spae1l5*abb254(11)
      abb254(42)=spae1l4*spbe2e1*spae2l5
      abb254(41)=abb254(42)+abb254(41)
      abb254(41)=-abb254(41)*abb254(7)*abb254(1)**4
      abb254(42)=-abb254(1)*abb254(7)
      abb254(25)=abb254(25)*abb254(42)
      abb254(43)=abb254(27)*abb254(25)
      abb254(35)=abb254(35)*spbl3k2
      abb254(44)=-abb254(43)*abb254(35)
      abb254(18)=abb254(32)+abb254(34)+abb254(44)+abb254(28)+abb254(24)+abb254(&
      &41)+abb254(18)
      abb254(24)=spae1l5*abb254(14)
      abb254(28)=spbe2e1*abb254(36)
      abb254(24)=abb254(24)-abb254(28)
      abb254(28)=spbk2e1*abb254(43)
      abb254(24)=abb254(28)-3.0_ki*abb254(24)
      abb254(9)=abb254(42)*abb254(9)
      abb254(27)=abb254(27)*abb254(9)
      abb254(28)=2.0_ki*spae1k1
      abb254(32)=abb254(28)*abb254(27)
      abb254(21)=abb254(42)*abb254(21)
      abb254(34)=spae1k1*abb254(21)
      abb254(41)=-abb254(30)*abb254(34)
      abb254(32)=abb254(32)+abb254(41)
      abb254(32)=spbk2k1*abb254(32)
      abb254(22)=-spae1l5*abb254(22)
      abb254(41)=abb254(27)*abb254(15)
      abb254(44)=spbl3k1*spal3l5
      abb254(45)=-abb254(34)*abb254(44)
      abb254(22)=abb254(45)+abb254(41)+abb254(22)+abb254(32)
      abb254(32)=-spae1l5*abb254(21)
      abb254(10)=-spae2l4*abb254(10)
      abb254(41)=-spak2l4*abb254(16)
      abb254(13)=abb254(13)*spae2l4
      abb254(45)=spak2l5*abb254(13)
      abb254(10)=abb254(45)+abb254(10)+abb254(41)
      abb254(10)=spbk2e1*abb254(10)
      abb254(41)=abb254(7)*spae2l5
      abb254(45)=abb254(41)*spak2l4
      abb254(7)=abb254(7)*spae2l4
      abb254(46)=abb254(7)*spak2l5
      abb254(45)=abb254(45)-abb254(46)
      abb254(46)=-spbk1e1*abb254(45)
      abb254(31)=abb254(9)*abb254(31)
      abb254(31)=abb254(31)+abb254(46)
      abb254(46)=spak1l3*spbl3k2*abb254(31)
      abb254(10)=abb254(10)+abb254(46)
      abb254(46)=abb254(9)*spae2l4
      abb254(45)=-abb254(46)+abb254(45)
      abb254(46)=spbk2e1*abb254(45)
      abb254(47)=abb254(7)*spbk1e1
      abb254(48)=spak1l5*abb254(47)
      abb254(49)=abb254(41)*spbk1e1
      abb254(50)=-spak1l4*abb254(49)
      abb254(46)=abb254(50)+abb254(46)+abb254(48)
      abb254(48)=spbl3e2*spae1l3
      abb254(50)=abb254(28)*spbe2k1
      abb254(48)=abb254(48)+abb254(50)
      abb254(50)=-abb254(13)*abb254(48)
      abb254(38)=abb254(38)*spae2l4
      abb254(51)=-spae1k2*abb254(38)
      abb254(23)=abb254(51)+2.0_ki*abb254(23)+abb254(50)
      abb254(50)=-3.0_ki*abb254(13)
      abb254(26)=-spbk2e1*abb254(26)
      abb254(35)=abb254(25)*abb254(35)
      abb254(26)=abb254(26)+abb254(35)
      abb254(35)=-spbk2e1*abb254(25)
      abb254(51)=-spbk2k1*abb254(28)
      abb254(15)=abb254(51)-abb254(15)
      abb254(15)=abb254(9)*abb254(15)
      abb254(51)=spae1k2*abb254(39)
      abb254(48)=abb254(51)+abb254(48)
      abb254(48)=abb254(16)*abb254(48)
      abb254(16)=3.0_ki*abb254(16)
      abb254(51)=-spal3l5*abb254(21)
      abb254(52)=abb254(43)*spbl3k2
      abb254(13)=abb254(13)*spae1l5
      abb254(13)=abb254(13)-abb254(36)
      abb254(53)=spbl3e2*abb254(13)
      abb254(52)=-abb254(52)+abb254(53)
      abb254(53)=-spbl3k2*abb254(45)
      abb254(54)=spbl3k2*abb254(25)
      abb254(55)=-spbk2k1*abb254(30)
      abb254(44)=-abb254(44)+abb254(55)
      abb254(42)=abb254(42)*abb254(20)
      abb254(55)=abb254(42)*spbe2e1
      abb254(56)=abb254(55)*spae1k1
      abb254(44)=abb254(56)*abb254(44)
      abb254(8)=-spae1l5*spbe2e1*abb254(8)*abb254(20)
      abb254(8)=abb254(8)+abb254(44)
      abb254(20)=-spae1l5*abb254(55)
      abb254(44)=spbk2e1*abb254(30)
      abb254(19)=abb254(44)-abb254(19)
      abb254(19)=abb254(42)*abb254(19)
      abb254(42)=-2.0_ki*abb254(42)
      abb254(44)=-spal3l5*abb254(55)
      abb254(12)=2.0_ki*abb254(12)
      abb254(57)=-abb254(30)*abb254(21)
      abb254(27)=2.0_ki*abb254(27)+abb254(57)
      abb254(9)=-2.0_ki*abb254(9)
      abb254(30)=-abb254(55)*abb254(30)
      abb254(38)=spae1l5*abb254(38)
      abb254(36)=-abb254(36)*abb254(39)
      abb254(36)=abb254(38)+abb254(36)
      abb254(33)=abb254(33)-abb254(40)
      abb254(21)=spak1l5*abb254(21)
      abb254(38)=spak1l5*abb254(55)
      abb254(39)=-spbk2k1*abb254(43)
      abb254(13)=spbe2k1*abb254(13)
      abb254(13)=abb254(39)+abb254(13)
      abb254(39)=-spbk2k1*abb254(45)
      abb254(40)=spbk2k1*abb254(25)
      abb254(45)=-spae1l3*abb254(14)
      abb254(55)=spak1l3*abb254(47)
      abb254(11)=-spae1k2*abb254(11)*abb254(37)
      abb254(37)=abb254(29)*spak1k2
      abb254(47)=abb254(47)*abb254(37)
      abb254(57)=abb254(29)*abb254(7)
      abb254(14)=abb254(14)*abb254(28)
      abb254(58)=spae1l3*abb254(17)
      abb254(59)=-spak1l3*abb254(49)
      abb254(60)=spae1k2*abb254(29)*abb254(17)
      abb254(37)=-abb254(49)*abb254(37)
      abb254(29)=-abb254(29)*abb254(41)
      abb254(17)=-abb254(17)*abb254(28)
      abb254(28)=-spbk1e1*abb254(43)
      abb254(25)=spbk1e1*abb254(25)
      R2d254=0.0_ki
      rat2 = rat2 + R2d254
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='254' value='", &
          & R2d254, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd254h0
