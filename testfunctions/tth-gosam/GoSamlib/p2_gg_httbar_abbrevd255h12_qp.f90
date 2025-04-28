module     p2_gg_httbar_abbrevd255h12_qp
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_kinematics_qp, only: epstensor
   use p2_gg_httbar_globalsh12_qp
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
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_color_qp, only: TR
      use p2_gg_httbar_globalsl1_qp, only: epspow
      implicit none
      abb255(1)=sqrt(mT**2)
      abb255(2)=NC**(-1)
      abb255(3)=spak2l4**(-1)
      abb255(4)=spak2l5**(-1)
      abb255(5)=spak2l3**(-1)
      abb255(6)=spbl3k2**(-1)
      abb255(7)=c1*abb255(2)
      abb255(7)=abb255(7)-c3
      abb255(7)=abb255(7)*gs**4*i_*TR*e*gHT
      abb255(8)=-abb255(7)*abb255(1)**2
      abb255(9)=abb255(8)*spbl4l3
      abb255(10)=spbl5k2*abb255(9)
      abb255(11)=abb255(8)*spbl4k2
      abb255(12)=-spbl5l3*abb255(11)
      abb255(10)=abb255(10)+abb255(12)
      abb255(10)=spae1l3*abb255(10)
      abb255(12)=abb255(3)*mT
      abb255(13)=abb255(12)*spak2l3
      abb255(14)=-abb255(1)*abb255(7)
      abb255(15)=abb255(13)*abb255(14)
      abb255(16)=spbl5k1*spae1k1
      abb255(17)=abb255(16)*abb255(15)
      abb255(18)=spbl3k2*abb255(17)
      abb255(10)=abb255(18)+abb255(10)
      abb255(18)=spbe2e1*spae2k2
      abb255(10)=abb255(18)*abb255(10)
      abb255(19)=abb255(18)*abb255(12)
      abb255(20)=abb255(16)*abb255(19)
      abb255(21)=abb255(4)*mT
      abb255(22)=abb255(21)*spae1e2
      abb255(23)=abb255(22)*spbl4e2
      abb255(24)=spak1k2*spbk1e1
      abb255(25)=-abb255(24)*abb255(23)
      abb255(25)=-abb255(20)+abb255(25)
      abb255(26)=-abb255(7)*abb255(1)**3
      abb255(25)=abb255(26)*abb255(25)
      abb255(27)=mH**2*abb255(6)*abb255(5)
      abb255(28)=abb255(27)*spbl4k2
      abb255(29)=-abb255(28)+2.0_ki*spbl4k2
      abb255(30)=-abb255(18)*abb255(29)
      abb255(31)=spae2l3*spbl4l3
      abb255(32)=spbe2e1*abb255(31)
      abb255(30)=abb255(32)+abb255(30)
      abb255(32)=abb255(21)*spae1k2
      abb255(33)=abb255(26)*abb255(32)
      abb255(30)=abb255(33)*abb255(30)
      abb255(34)=abb255(27)*abb255(8)
      abb255(35)=spbl4e2*spae1e2
      abb255(36)=abb255(34)*abb255(35)
      abb255(37)=abb255(8)*abb255(35)
      abb255(38)=2.0_ki*abb255(37)
      abb255(36)=abb255(36)-abb255(38)
      abb255(36)=abb255(36)*spbl5k2
      abb255(39)=-abb255(24)*abb255(36)
      abb255(40)=abb255(28)*abb255(8)
      abb255(41)=2.0_ki*abb255(11)
      abb255(40)=abb255(40)-abb255(41)
      abb255(40)=spae1e2*abb255(40)
      abb255(42)=abb255(24)*abb255(40)
      abb255(43)=abb255(7)*abb255(1)**4
      abb255(44)=spbl4e1*spae1e2*abb255(43)
      abb255(42)=abb255(44)+abb255(42)
      abb255(42)=spbl5e2*abb255(42)
      abb255(35)=-spbl5e1*abb255(43)*abb255(35)
      abb255(22)=abb255(14)*abb255(22)
      abb255(43)=abb255(22)*abb255(24)
      abb255(44)=spbl4l3*spbk2e2*spak2l3
      abb255(45)=-abb255(43)*abb255(44)
      abb255(46)=abb255(37)*spbl5l3
      abb255(47)=abb255(8)*spbl5e2
      abb255(48)=abb255(47)*spae1e2
      abb255(49)=abb255(48)*spbl4l3
      abb255(46)=abb255(46)-abb255(49)
      abb255(49)=spak1l3*spbk1e1
      abb255(50)=-abb255(46)*abb255(49)
      abb255(51)=abb255(26)*spbl5e1
      abb255(13)=-spbl3e2*abb255(13)*abb255(51)*spae1e2
      abb255(10)=abb255(13)+abb255(50)+abb255(45)+abb255(35)+abb255(39)+abb255(&
      &30)+abb255(25)+abb255(42)+abb255(10)
      abb255(13)=2.0_ki*abb255(14)
      abb255(25)=-abb255(13)*abb255(20)
      abb255(30)=spbl5e1*abb255(37)
      abb255(35)=spbl4e1*abb255(48)
      abb255(30)=abb255(30)-abb255(35)
      abb255(35)=-spbl4e2*abb255(43)
      abb255(20)=-abb255(14)*abb255(20)
      abb255(20)=abb255(20)+abb255(35)+3.0_ki*abb255(30)
      abb255(30)=spae1k2*abb255(34)
      abb255(35)=abb255(8)*spae1k2
      abb255(30)=-2.0_ki*abb255(35)+abb255(30)
      abb255(30)=spbl5k2*abb255(30)
      abb255(30)=abb255(33)+abb255(30)
      abb255(30)=spbl4e2*abb255(30)
      abb255(37)=abb255(35)*abb255(28)
      abb255(39)=spae1k2*abb255(41)
      abb255(39)=abb255(39)-abb255(37)
      abb255(39)=spbl5e2*abb255(39)
      abb255(41)=abb255(32)*abb255(14)
      abb255(42)=abb255(41)*abb255(44)
      abb255(45)=-spbl4l3*abb255(47)
      abb255(50)=abb255(8)*spbl4e2
      abb255(52)=spbl5l3*abb255(50)
      abb255(45)=abb255(45)+abb255(52)
      abb255(45)=spae1l3*abb255(45)
      abb255(52)=-spbl3e2*abb255(17)
      abb255(30)=abb255(52)+abb255(45)+abb255(42)+abb255(39)+abb255(30)
      abb255(39)=spbl4e2*abb255(41)
      abb255(42)=abb255(14)*abb255(21)
      abb255(45)=abb255(24)*abb255(42)
      abb255(29)=-abb255(45)*abb255(29)
      abb255(51)=abb255(51)*abb255(12)
      abb255(52)=abb255(15)*spbl3k2
      abb255(53)=-spbl5e1*abb255(52)
      abb255(29)=abb255(53)+abb255(51)+abb255(29)
      abb255(29)=spae2k2*abb255(29)
      abb255(53)=abb255(7)*spae2k2
      abb255(54)=abb255(53)*spbl5k2
      abb255(55)=abb255(54)*spbl4l3
      abb255(53)=abb255(53)*spbl4k2
      abb255(56)=abb255(53)*spbl5l3
      abb255(55)=abb255(55)-abb255(56)
      abb255(49)=-abb255(55)*abb255(49)
      abb255(56)=abb255(45)*abb255(31)
      abb255(29)=abb255(56)+abb255(49)+abb255(29)
      abb255(49)=abb255(13)*abb255(12)
      abb255(56)=spbl5e1*spae2k2*abb255(49)
      abb255(12)=spae2k2*abb255(14)*abb255(12)
      abb255(12)=-abb255(53)+abb255(12)
      abb255(12)=spbl5e1*abb255(12)
      abb255(57)=spbl4e1*abb255(54)
      abb255(12)=abb255(12)+abb255(57)
      abb255(9)=-spae1l3*abb255(9)
      abb255(9)=abb255(9)-abb255(37)
      abb255(9)=spbe2e1*abb255(9)
      abb255(37)=spae1k2*spbe2e1
      abb255(57)=abb255(11)*abb255(37)
      abb255(9)=abb255(57)+abb255(9)
      abb255(57)=abb255(7)*spbl4k2
      abb255(58)=-abb255(24)*abb255(57)
      abb255(24)=abb255(7)*abb255(24)
      abb255(59)=abb255(28)*abb255(24)
      abb255(60)=abb255(7)*spbk1e1
      abb255(61)=abb255(60)*spak1l3
      abb255(62)=spbl4l3*abb255(61)
      abb255(58)=abb255(62)+abb255(58)+abb255(59)
      abb255(11)=abb255(52)-abb255(11)
      abb255(11)=abb255(18)*abb255(11)
      abb255(52)=-abb255(26)*abb255(19)
      abb255(59)=spak1e2*spbk1e1
      abb255(62)=abb255(50)*abb255(59)
      abb255(11)=abb255(62)+abb255(52)+abb255(11)
      abb255(52)=-abb255(19)*abb255(13)
      abb255(19)=-abb255(14)*abb255(19)
      abb255(60)=abb255(60)*spak1e2
      abb255(62)=-spbl4e2*abb255(60)
      abb255(19)=abb255(19)+abb255(62)
      abb255(62)=-spbl3e2*abb255(15)
      abb255(50)=3.0_ki*abb255(50)+abb255(62)
      abb255(62)=-spbl4e2*abb255(7)
      abb255(63)=spae1l3*spbl5l3*abb255(8)
      abb255(33)=abb255(63)+2.0_ki*abb255(33)
      abb255(33)=spbe2e1*abb255(33)
      abb255(34)=abb255(34)*abb255(37)
      abb255(35)=abb255(35)*spbe2e1
      abb255(34)=-abb255(35)+abb255(34)
      abb255(34)=spbl5k2*abb255(34)
      abb255(33)=abb255(34)+abb255(33)
      abb255(27)=abb255(27)-1.0_ki
      abb255(27)=abb255(27)*spbl5k2
      abb255(24)=-abb255(24)*abb255(27)
      abb255(34)=-spbl5l3*abb255(61)
      abb255(24)=abb255(34)+2.0_ki*abb255(45)+abb255(24)
      abb255(8)=spbl5k2*abb255(8)*abb255(18)
      abb255(18)=-abb255(47)*abb255(59)
      abb255(8)=abb255(8)+abb255(18)
      abb255(18)=spbl5e2*abb255(60)
      abb255(34)=-3.0_ki*abb255(47)
      abb255(35)=spbl5e2*abb255(7)
      abb255(17)=-spbe2e1*abb255(17)
      abb255(37)=spbl5e1*abb255(15)
      abb255(15)=-spbe2e1*abb255(15)
      abb255(45)=spbl4l3*abb255(43)
      abb255(47)=-spbl4l3*abb255(41)
      abb255(59)=spbl4l3*abb255(7)
      abb255(60)=-spbl5l3*abb255(7)
      abb255(61)=abb255(28)-spbl4k2
      abb255(63)=abb255(43)*abb255(61)
      abb255(64)=2.0_ki*spae1e2
      abb255(51)=abb255(64)*abb255(51)
      abb255(51)=abb255(51)+abb255(63)
      abb255(63)=abb255(14)*spbl4k2
      abb255(32)=abb255(32)*abb255(63)
      abb255(64)=-abb255(28)*abb255(41)
      abb255(16)=abb255(16)*abb255(49)
      abb255(16)=abb255(16)+abb255(32)+abb255(64)
      abb255(23)=-abb255(26)*abb255(23)
      abb255(26)=spbl5e2*abb255(40)
      abb255(32)=-abb255(22)*abb255(44)
      abb255(23)=abb255(32)+abb255(26)+abb255(23)-abb255(36)
      abb255(26)=-spbl4e2*abb255(22)
      abb255(14)=abb255(14)*abb255(28)
      abb255(14)=-2.0_ki*abb255(63)+abb255(14)
      abb255(14)=abb255(14)*abb255(21)*spae2k2
      abb255(31)=abb255(31)*abb255(42)
      abb255(14)=abb255(31)+abb255(14)
      abb255(28)=abb255(7)*abb255(28)
      abb255(28)=-abb255(57)+abb255(28)
      abb255(13)=abb255(21)*abb255(13)
      abb255(7)=-abb255(7)*abb255(27)
      abb255(7)=abb255(13)+abb255(7)
      abb255(13)=spbl4l3*abb255(22)
      abb255(21)=abb255(22)*abb255(61)
      abb255(27)=-spbk2e2*abb255(43)
      abb255(31)=spbk2e2*abb255(41)
      abb255(22)=-spbk2e2*abb255(22)
      abb255(32)=-spbk1e1*abb255(38)
      abb255(36)=abb255(53)*spbk1e1
      abb255(38)=2.0_ki*spbk1e1*abb255(48)
      abb255(40)=-spbk1e1*abb255(54)
      R2d255=0.0_ki
      rat2 = rat2 + R2d255
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='255' value='", &
          & R2d255, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd255h12_qp
