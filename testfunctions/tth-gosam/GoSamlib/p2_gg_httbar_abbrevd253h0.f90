module     p2_gg_httbar_abbrevd253h0
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_kinematics, only: epstensor
   use p2_gg_httbar_globalsh0
   implicit none
   private
   complex(ki), dimension(61), public :: abb253
   complex(ki), public :: R2d253
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
      abb253(1)=sqrt(mT**2)
      abb253(2)=NC**(-1)
      abb253(3)=spbl4k2**(-1)
      abb253(4)=spak2l3**(-1)
      abb253(5)=spbl3k2**(-1)
      abb253(6)=spbl5k2**(-1)
      abb253(7)=mH**2*abb253(5)*abb253(4)
      abb253(8)=c2*abb253(2)
      abb253(8)=abb253(8)-c3
      abb253(8)=abb253(8)*gs**4*i_*TR*e*gHT
      abb253(9)=-abb253(8)*abb253(1)**2
      abb253(10)=abb253(7)*abb253(9)
      abb253(11)=spak1l4*spbk1e1
      abb253(12)=-abb253(11)*abb253(10)
      abb253(13)=-abb253(8)*abb253(1)**3
      abb253(14)=abb253(3)*mT
      abb253(15)=abb253(13)*abb253(14)
      abb253(16)=-spbk2e1*abb253(15)
      abb253(12)=abb253(16)+abb253(12)
      abb253(16)=spbk2e2*spae1e2
      abb253(17)=abb253(16)*spak2l5
      abb253(12)=abb253(17)*abb253(12)
      abb253(18)=spbl3e1*spal3l4
      abb253(18)=-abb253(18)+2.0_ki*abb253(11)
      abb253(19)=abb253(6)*mT
      abb253(20)=abb253(13)*abb253(19)
      abb253(21)=abb253(20)*abb253(16)
      abb253(22)=-abb253(21)*abb253(18)
      abb253(23)=abb253(9)*spae2k2
      abb253(24)=spae1l4*spbe2e1
      abb253(25)=abb253(23)*abb253(24)
      abb253(26)=-spal3l5*abb253(25)
      abb253(27)=abb253(14)*spae2l5
      abb253(28)=abb253(13)*abb253(27)
      abb253(29)=abb253(28)*spbe2e1
      abb253(30)=-spae1l3*abb253(29)
      abb253(31)=abb253(9)*spae1l5
      abb253(32)=abb253(31)*spbe2e1
      abb253(33)=abb253(32)*spae2k2
      abb253(34)=spal3l4*abb253(33)
      abb253(26)=abb253(30)+abb253(26)+abb253(34)
      abb253(26)=spbl3k2*abb253(26)
      abb253(30)=abb253(10)*abb253(16)
      abb253(34)=spak1l5*spbk1e1
      abb253(35)=abb253(34)*abb253(30)
      abb253(36)=spbk2e1*abb253(7)*abb253(21)
      abb253(35)=abb253(35)+abb253(36)
      abb253(35)=spak2l4*abb253(35)
      abb253(36)=-spae2l5*abb253(24)
      abb253(37)=spae2l4*spae1l5*spbe2e1
      abb253(36)=abb253(37)+abb253(36)
      abb253(36)=abb253(36)*abb253(8)*abb253(1)**4
      abb253(37)=-spak1l5*abb253(25)
      abb253(38)=spak1l4*abb253(33)
      abb253(37)=abb253(38)+abb253(37)
      abb253(37)=spbk2k1*abb253(37)
      abb253(29)=2.0_ki*abb253(29)
      abb253(38)=spae1k1*spbk2k1
      abb253(39)=-abb253(38)*abb253(29)
      abb253(40)=abb253(11)*spal3l5
      abb253(41)=abb253(34)*spal3l4
      abb253(40)=abb253(40)-abb253(41)
      abb253(41)=abb253(9)*spae1e2
      abb253(42)=-spbl3e2*abb253(41)*abb253(40)
      abb253(43)=-abb253(1)*abb253(8)
      abb253(44)=abb253(43)*abb253(14)*spbl3k2
      abb253(45)=spak2l5*abb253(44)
      abb253(16)=abb253(45)*abb253(16)
      abb253(46)=spak1l3*spbk1e1
      abb253(47)=abb253(46)*abb253(16)
      abb253(12)=abb253(47)+abb253(42)+abb253(39)+2.0_ki*abb253(37)+abb253(35)+&
      &abb253(26)+abb253(22)+abb253(12)+abb253(36)
      abb253(22)=abb253(24)*abb253(9)
      abb253(26)=spae2l5*abb253(22)
      abb253(35)=spae2l4*abb253(32)
      abb253(26)=abb253(26)-abb253(35)
      abb253(14)=abb253(43)*abb253(14)
      abb253(35)=abb253(17)*abb253(14)
      abb253(36)=-spbk2e1*abb253(35)
      abb253(26)=abb253(36)+3.0_ki*abb253(26)
      abb253(36)=abb253(10)*spak2l5
      abb253(37)=spae1l4*abb253(36)
      abb253(39)=abb253(7)*spak2l4
      abb253(42)=-abb253(31)*abb253(39)
      abb253(45)=-spae1l3*abb253(45)
      abb253(37)=abb253(45)+abb253(42)+abb253(37)
      abb253(37)=spbk2e2*abb253(37)
      abb253(42)=abb253(19)*spbk2e2
      abb253(45)=abb253(43)*abb253(42)
      abb253(47)=abb253(45)*abb253(39)
      abb253(48)=2.0_ki*abb253(14)
      abb253(49)=spbk2e2*abb253(48)*spak2l5
      abb253(47)=abb253(47)-abb253(49)
      abb253(49)=abb253(47)*abb253(38)
      abb253(13)=spae1l4*abb253(13)*abb253(42)
      abb253(42)=-spal3l4*abb253(31)
      abb253(9)=abb253(9)*spae1l4
      abb253(50)=spal3l5*abb253(9)
      abb253(42)=abb253(42)+abb253(50)
      abb253(42)=spbl3e2*abb253(42)
      abb253(50)=-spak1l4*abb253(31)
      abb253(51)=spak1l5*abb253(9)
      abb253(50)=abb253(50)+abb253(51)
      abb253(50)=spbe2k1*abb253(50)
      abb253(51)=abb253(45)*spae1k1
      abb253(52)=spbl3k1*spal3l4
      abb253(53)=abb253(51)*abb253(52)
      abb253(13)=abb253(53)+2.0_ki*abb253(50)+abb253(42)+abb253(49)+abb253(13)+&
      &abb253(37)
      abb253(37)=spae1l4*abb253(45)
      abb253(42)=abb253(8)*spae2k2
      abb253(49)=abb253(42)*spbl3k2
      abb253(40)=abb253(49)*abb253(40)
      abb253(28)=spbk2e1*abb253(28)
      abb253(27)=abb253(27)*abb253(43)
      abb253(50)=abb253(27)*spbl3k2
      abb253(53)=-abb253(46)*abb253(50)
      abb253(28)=abb253(53)+abb253(28)+abb253(40)
      abb253(40)=spbk2e1*abb253(27)
      abb253(53)=abb253(8)*spbk1e1
      abb253(54)=abb253(53)*spak1l4
      abb253(55)=-spae2l5*abb253(54)
      abb253(53)=abb253(53)*spak1l5
      abb253(56)=spae2l4*abb253(53)
      abb253(40)=abb253(56)+abb253(55)+abb253(40)
      abb253(55)=abb253(11)*abb253(41)
      abb253(15)=spae1e2*abb253(15)
      abb253(23)=-spae1l4*abb253(23)
      abb253(15)=abb253(15)+abb253(23)
      abb253(15)=spbk2e1*abb253(15)
      abb253(23)=abb253(44)*spae1e2
      abb253(46)=-abb253(46)*abb253(23)
      abb253(15)=abb253(46)+abb253(55)+abb253(15)
      abb253(14)=abb253(14)*spae1e2
      abb253(46)=spae1l4*abb253(42)
      abb253(46)=abb253(14)+abb253(46)
      abb253(46)=spbk2e1*abb253(46)
      abb253(55)=abb253(38)*abb253(48)
      abb253(44)=spae1l3*abb253(44)
      abb253(9)=abb253(44)-3.0_ki*abb253(9)+abb253(55)
      abb253(44)=spae1l4*abb253(8)
      abb253(55)=spbl3e2*spal3l4
      abb253(56)=spbe2k1*spak1l4
      abb253(55)=abb253(56)+abb253(55)
      abb253(55)=abb253(41)*abb253(55)
      abb253(30)=spak2l4*abb253(30)
      abb253(30)=abb253(30)+abb253(55)
      abb253(55)=-spal3l4*abb253(49)
      abb253(56)=abb253(42)*spbk2k1
      abb253(57)=-spak1l4*abb253(56)
      abb253(55)=abb253(55)+abb253(57)
      abb253(57)=-abb253(41)*abb253(34)
      abb253(58)=spbk2e1*abb253(31)*spae2k2
      abb253(57)=abb253(57)+abb253(58)
      abb253(58)=-spbk2e1*spae1l5*abb253(42)
      abb253(31)=3.0_ki*abb253(31)
      abb253(59)=-spae1l5*abb253(8)
      abb253(60)=-spbl3e2*spal3l5
      abb253(61)=-spbe2k1*spak1l5
      abb253(60)=abb253(61)+abb253(60)
      abb253(41)=abb253(41)*abb253(60)
      abb253(10)=-abb253(10)*abb253(17)
      abb253(10)=-2.0_ki*abb253(21)+abb253(10)+abb253(41)
      abb253(17)=spak1l5*abb253(56)
      abb253(21)=spal3l5*abb253(49)
      abb253(17)=abb253(17)+abb253(21)
      abb253(21)=spal3l5*abb253(22)
      abb253(41)=-spal3l4*abb253(32)
      abb253(21)=abb253(41)+abb253(21)
      abb253(41)=spal3l4*abb253(53)
      abb253(49)=-spal3l5*abb253(54)
      abb253(41)=abb253(41)+abb253(49)
      abb253(49)=spal3l4*abb253(8)
      abb253(56)=-spal3l5*abb253(8)
      abb253(60)=abb253(45)*spal3l4
      abb253(20)=abb253(20)+abb253(36)
      abb253(20)=abb253(24)*abb253(20)
      abb253(19)=abb253(43)*abb253(19)
      abb253(36)=abb253(19)*spbe2e1
      abb253(38)=abb253(36)*abb253(38)
      abb253(38)=abb253(38)-abb253(32)
      abb253(38)=abb253(39)*abb253(38)
      abb253(43)=abb253(36)*spae1k1
      abb253(52)=abb253(43)*abb253(52)
      abb253(20)=abb253(52)+abb253(38)+abb253(20)
      abb253(24)=abb253(24)*abb253(19)
      abb253(18)=abb253(19)*abb253(18)
      abb253(7)=abb253(7)*spak2l5
      abb253(38)=-abb253(54)*abb253(7)
      abb253(52)=-spbk2e1*abb253(19)
      abb253(52)=abb253(53)+abb253(52)
      abb253(52)=abb253(52)*abb253(39)
      abb253(18)=abb253(52)+abb253(38)+abb253(18)
      abb253(38)=abb253(8)*abb253(39)
      abb253(7)=-abb253(8)*abb253(7)
      abb253(7)=2.0_ki*abb253(19)+abb253(7)
      abb253(19)=spal3l4*abb253(36)
      abb253(39)=abb253(39)*abb253(36)
      abb253(22)=-spak1l5*abb253(22)
      abb253(32)=spak1l4*abb253(32)
      abb253(22)=abb253(32)+abb253(22)
      abb253(22)=2.0_ki*abb253(22)
      abb253(32)=-spak1l4*abb253(8)
      abb253(8)=spak1l5*abb253(8)
      abb253(45)=-spak1l4*abb253(45)
      abb253(36)=-spak1l4*abb253(36)
      abb253(52)=spbk2k1*abb253(35)
      abb253(53)=-spbk2k1*abb253(27)
      abb253(54)=-spbk2k1*abb253(14)
      abb253(25)=2.0_ki*abb253(25)
      abb253(11)=-abb253(42)*abb253(11)
      abb253(33)=-2.0_ki*abb253(33)
      abb253(34)=abb253(42)*abb253(34)
      abb253(35)=spbk1e1*abb253(35)
      abb253(27)=-spbk1e1*abb253(27)
      abb253(14)=-spbk1e1*abb253(14)
      R2d253=0.0_ki
      rat2 = rat2 + R2d253
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='253' value='", &
          & R2d253, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd253h0
