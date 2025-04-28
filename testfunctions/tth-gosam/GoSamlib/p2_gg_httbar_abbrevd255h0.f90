module     p2_gg_httbar_abbrevd255h0
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_kinematics, only: epstensor
   use p2_gg_httbar_globalsh0
   implicit none
   private
   complex(ki), dimension(64), public :: abb255
   complex(ki), public :: R2d255
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
      abb255(1)=sqrt(mT**2)
      abb255(2)=NC**(-1)
      abb255(3)=spbl4k2**(-1)
      abb255(4)=spbl5k2**(-1)
      abb255(5)=spak2l3**(-1)
      abb255(6)=spbl3k2**(-1)
      abb255(7)=c1*abb255(2)
      abb255(7)=abb255(7)-c3
      abb255(7)=abb255(7)*gs**4*i_*TR*e*gHT
      abb255(8)=-abb255(7)*abb255(1)**2
      abb255(9)=abb255(8)*spal3l4
      abb255(10)=-spak2l5*abb255(9)
      abb255(11)=abb255(8)*spak2l4
      abb255(12)=spal3l5*abb255(11)
      abb255(10)=abb255(10)+abb255(12)
      abb255(10)=spbl3e1*abb255(10)
      abb255(12)=abb255(3)*mT
      abb255(13)=abb255(12)*spbl3k2
      abb255(14)=-abb255(1)*abb255(7)
      abb255(15)=abb255(13)*abb255(14)
      abb255(16)=spak1l5*spbk1e1
      abb255(17)=abb255(16)*abb255(15)
      abb255(18)=-spak2l3*abb255(17)
      abb255(10)=abb255(18)+abb255(10)
      abb255(18)=spae1e2*spbk2e2
      abb255(10)=abb255(18)*abb255(10)
      abb255(19)=abb255(18)*abb255(12)
      abb255(20)=abb255(16)*abb255(19)
      abb255(21)=abb255(4)*mT
      abb255(22)=abb255(21)*spbe2e1
      abb255(23)=abb255(22)*spae2l4
      abb255(24)=spbk2k1*spae1k1
      abb255(25)=abb255(24)*abb255(23)
      abb255(25)=abb255(20)+abb255(25)
      abb255(26)=-abb255(7)*abb255(1)**3
      abb255(25)=abb255(26)*abb255(25)
      abb255(27)=mH**2*abb255(6)*abb255(5)
      abb255(28)=abb255(27)*spak2l4
      abb255(29)=-abb255(28)+2.0_ki*spak2l4
      abb255(30)=abb255(18)*abb255(29)
      abb255(31)=spbl3e2*spal3l4
      abb255(32)=-spae1e2*abb255(31)
      abb255(30)=abb255(32)+abb255(30)
      abb255(32)=abb255(21)*spbk2e1
      abb255(33)=abb255(26)*abb255(32)
      abb255(30)=abb255(33)*abb255(30)
      abb255(34)=abb255(27)*abb255(8)
      abb255(35)=spae2l4*spbe2e1
      abb255(36)=abb255(34)*abb255(35)
      abb255(37)=abb255(8)*abb255(35)
      abb255(38)=2.0_ki*abb255(37)
      abb255(36)=abb255(36)-abb255(38)
      abb255(36)=abb255(36)*spak2l5
      abb255(39)=abb255(24)*abb255(36)
      abb255(40)=abb255(28)*abb255(8)
      abb255(41)=2.0_ki*abb255(11)
      abb255(40)=abb255(40)-abb255(41)
      abb255(40)=spbe2e1*abb255(40)
      abb255(42)=-abb255(24)*abb255(40)
      abb255(43)=abb255(7)*abb255(1)**4
      abb255(44)=-spae1l4*spbe2e1*abb255(43)
      abb255(42)=abb255(44)+abb255(42)
      abb255(42)=spae2l5*abb255(42)
      abb255(35)=spae1l5*abb255(43)*abb255(35)
      abb255(22)=abb255(14)*abb255(22)
      abb255(43)=abb255(22)*abb255(24)
      abb255(44)=spal3l4*spae2k2*spbl3k2
      abb255(45)=abb255(43)*abb255(44)
      abb255(46)=abb255(37)*spal3l5
      abb255(47)=abb255(8)*spae2l5
      abb255(48)=abb255(47)*spbe2e1
      abb255(49)=abb255(48)*spal3l4
      abb255(46)=abb255(46)-abb255(49)
      abb255(49)=spbl3k1*spae1k1
      abb255(50)=abb255(46)*abb255(49)
      abb255(51)=abb255(26)*spae1l5
      abb255(13)=spae2l3*abb255(13)*abb255(51)*spbe2e1
      abb255(10)=abb255(13)+abb255(50)+abb255(45)+abb255(35)+abb255(39)+abb255(&
      &30)+abb255(25)+abb255(42)+abb255(10)
      abb255(13)=2.0_ki*abb255(14)
      abb255(25)=abb255(13)*abb255(20)
      abb255(30)=spae1l5*abb255(37)
      abb255(35)=spae1l4*abb255(48)
      abb255(30)=abb255(30)-abb255(35)
      abb255(35)=spae2l4*abb255(43)
      abb255(20)=abb255(14)*abb255(20)
      abb255(20)=abb255(20)+abb255(35)-3.0_ki*abb255(30)
      abb255(30)=abb255(14)*abb255(21)
      abb255(35)=abb255(24)*abb255(30)
      abb255(29)=abb255(35)*abb255(29)
      abb255(37)=abb255(51)*abb255(12)
      abb255(39)=abb255(15)*spak2l3
      abb255(42)=spae1l5*abb255(39)
      abb255(29)=abb255(42)-abb255(37)+abb255(29)
      abb255(29)=spbk2e2*abb255(29)
      abb255(42)=abb255(7)*spbk2e2
      abb255(45)=abb255(42)*spak2l5
      abb255(50)=abb255(45)*spal3l4
      abb255(42)=abb255(42)*spak2l4
      abb255(51)=abb255(42)*spal3l5
      abb255(50)=abb255(50)-abb255(51)
      abb255(49)=abb255(50)*abb255(49)
      abb255(51)=-abb255(35)*abb255(31)
      abb255(29)=abb255(51)+abb255(49)+abb255(29)
      abb255(49)=abb255(13)*abb255(12)
      abb255(51)=-spae1l5*spbk2e2*abb255(49)
      abb255(12)=-spbk2e2*abb255(14)*abb255(12)
      abb255(12)=abb255(42)+abb255(12)
      abb255(12)=spae1l5*abb255(12)
      abb255(52)=-spae1l4*abb255(45)
      abb255(12)=abb255(12)+abb255(52)
      abb255(52)=-spbk2e1*abb255(34)
      abb255(53)=abb255(8)*spbk2e1
      abb255(52)=2.0_ki*abb255(53)+abb255(52)
      abb255(52)=spak2l5*abb255(52)
      abb255(52)=-abb255(33)+abb255(52)
      abb255(52)=spae2l4*abb255(52)
      abb255(54)=abb255(53)*abb255(28)
      abb255(41)=-spbk2e1*abb255(41)
      abb255(41)=abb255(41)+abb255(54)
      abb255(41)=spae2l5*abb255(41)
      abb255(55)=abb255(32)*abb255(14)
      abb255(56)=-abb255(55)*abb255(44)
      abb255(57)=spal3l4*abb255(47)
      abb255(58)=abb255(8)*spae2l4
      abb255(59)=-spal3l5*abb255(58)
      abb255(57)=abb255(57)+abb255(59)
      abb255(57)=spbl3e1*abb255(57)
      abb255(59)=spae2l3*abb255(17)
      abb255(41)=abb255(59)+abb255(57)+abb255(56)+abb255(41)+abb255(52)
      abb255(52)=-spae2l4*abb255(55)
      abb255(9)=spbl3e1*abb255(9)
      abb255(9)=abb255(9)+abb255(54)
      abb255(9)=spae1e2*abb255(9)
      abb255(54)=spbk2e1*spae1e2
      abb255(56)=-abb255(11)*abb255(54)
      abb255(9)=abb255(56)+abb255(9)
      abb255(56)=abb255(7)*spak2l4
      abb255(57)=abb255(24)*abb255(56)
      abb255(24)=abb255(7)*abb255(24)
      abb255(59)=-abb255(28)*abb255(24)
      abb255(60)=abb255(7)*spae1k1
      abb255(61)=abb255(60)*spbl3k1
      abb255(62)=-spal3l4*abb255(61)
      abb255(57)=abb255(62)+abb255(57)+abb255(59)
      abb255(11)=-abb255(39)+abb255(11)
      abb255(11)=abb255(18)*abb255(11)
      abb255(39)=abb255(26)*abb255(19)
      abb255(59)=spbe2k1*spae1k1
      abb255(62)=-abb255(58)*abb255(59)
      abb255(11)=abb255(62)+abb255(39)+abb255(11)
      abb255(39)=abb255(19)*abb255(13)
      abb255(19)=abb255(14)*abb255(19)
      abb255(60)=abb255(60)*spbe2k1
      abb255(62)=spae2l4*abb255(60)
      abb255(19)=abb255(19)+abb255(62)
      abb255(62)=spae2l3*abb255(15)
      abb255(58)=-3.0_ki*abb255(58)+abb255(62)
      abb255(62)=spae2l4*abb255(7)
      abb255(63)=-spbl3e1*spal3l5*abb255(8)
      abb255(33)=abb255(63)-2.0_ki*abb255(33)
      abb255(33)=spae1e2*abb255(33)
      abb255(34)=-abb255(34)*abb255(54)
      abb255(53)=abb255(53)*spae1e2
      abb255(34)=abb255(53)+abb255(34)
      abb255(34)=spak2l5*abb255(34)
      abb255(33)=abb255(34)+abb255(33)
      abb255(27)=abb255(27)-1.0_ki
      abb255(27)=abb255(27)*spak2l5
      abb255(24)=abb255(24)*abb255(27)
      abb255(34)=spal3l5*abb255(61)
      abb255(24)=abb255(34)-2.0_ki*abb255(35)+abb255(24)
      abb255(8)=-spak2l5*abb255(8)*abb255(18)
      abb255(18)=abb255(47)*abb255(59)
      abb255(8)=abb255(8)+abb255(18)
      abb255(18)=-spae2l5*abb255(60)
      abb255(34)=3.0_ki*abb255(47)
      abb255(35)=-spae2l5*abb255(7)
      abb255(47)=-spal3l4*abb255(43)
      abb255(53)=spal3l4*abb255(55)
      abb255(17)=spae1e2*abb255(17)
      abb255(54)=-spae1l5*abb255(15)
      abb255(15)=spae1e2*abb255(15)
      abb255(59)=-spal3l4*abb255(7)
      abb255(60)=spal3l5*abb255(7)
      abb255(61)=abb255(28)-spak2l4
      abb255(63)=-abb255(43)*abb255(61)
      abb255(64)=2.0_ki*spbe2e1
      abb255(37)=-abb255(64)*abb255(37)
      abb255(37)=abb255(37)+abb255(63)
      abb255(63)=abb255(14)*spak2l4
      abb255(32)=-abb255(32)*abb255(63)
      abb255(64)=abb255(28)*abb255(55)
      abb255(16)=-abb255(16)*abb255(49)
      abb255(16)=abb255(16)+abb255(32)+abb255(64)
      abb255(23)=abb255(26)*abb255(23)
      abb255(26)=-spae2l5*abb255(40)
      abb255(32)=abb255(22)*abb255(44)
      abb255(23)=abb255(32)+abb255(26)+abb255(23)+abb255(36)
      abb255(26)=spae2l4*abb255(22)
      abb255(14)=-abb255(14)*abb255(28)
      abb255(14)=2.0_ki*abb255(63)+abb255(14)
      abb255(14)=abb255(14)*abb255(21)*spbk2e2
      abb255(30)=-abb255(31)*abb255(30)
      abb255(14)=abb255(30)+abb255(14)
      abb255(28)=-abb255(7)*abb255(28)
      abb255(28)=abb255(56)+abb255(28)
      abb255(13)=-abb255(21)*abb255(13)
      abb255(7)=abb255(7)*abb255(27)
      abb255(7)=abb255(13)+abb255(7)
      abb255(13)=-spal3l4*abb255(22)
      abb255(21)=-abb255(22)*abb255(61)
      abb255(27)=spae1k1*abb255(38)
      abb255(30)=abb255(42)*spae1k1
      abb255(31)=spae2k2*abb255(43)
      abb255(32)=-spae2k2*abb255(55)
      abb255(22)=spae2k2*abb255(22)
      abb255(36)=-2.0_ki*spae1k1*abb255(48)
      abb255(38)=spae1k1*abb255(45)
      R2d255=0.0_ki
      rat2 = rat2 + R2d255
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='255' value='", &
          & R2d255, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd255h0
